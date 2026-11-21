"""LoGIC-conditioned model for CFD prediction.

Tensors use B,C,X,Y,Z. Geometry channels are fluid fraction, signed distance,
2*x/size-1, 2*y/size-1, 2*z/size-1, and buffer indicator. Only channels 2:5
are used to locate the 3x3x3 LoGIC controls in physical specimen coordinates.
No solver result, label-validity mask, or sample ID is an input.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F


def encode_controls(arrays: dict) -> np.ndarray:
    """Return 13x3x3x3 controls in fixed G/D/P, density, row-major R order."""
    names = np.asarray(arrays["topology_names"]).tolist()
    ordered = ["Gyroid", "Diamond", "Primitive"]
    if sorted(names) != sorted(ordered):
        raise ValueError("Expected Gyroid/Diamond/Primitive topology names")
    weights = np.asarray(arrays["weights"], dtype=float)[..., [names.index(k) for k in ordered]]
    density = np.asarray(arrays["density"], dtype=float)
    q = np.asarray(arrays["quaternion_xyzw"], dtype=float)
    if weights.shape != (3, 3, 3, 3) or density.shape != (3, 3, 3) or q.shape != (3, 3, 3, 4):
        raise ValueError("Expected a 3x3x3 control lattice")
    if not all(np.isfinite(a).all() for a in (weights, density, q)):
        raise ValueError("Nonfinite controls")
    if (np.any(weights < 0) or not np.allclose(weights.sum(-1), 1, atol=1e-6)
            or np.any((density <= 0) | (density >= 1))):
        raise ValueError("Invalid weights or density")
    if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5):
        raise ValueError("Expected normalized xyzw quaternions")
    rotation = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(3, 3, 3, 9)
    controls = np.concatenate((weights, density[..., None], rotation), axis=-1)
    return np.moveaxis(controls, -1, 0).astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        groups = math.gcd(outputs, 8)
        self.body = nn.Sequential(
            nn.Conv3d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs), nn.SiLU(),
            nn.Conv3d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs),
        )
        self.skip = nn.Identity() if inputs == outputs else nn.Conv3d(inputs, outputs, 1)

    def forward(self, value):
        return F.silu(self.body(value) + self.skip(value))


class ExtraConvolution(nn.Module):
    """A small residual bottleneck used to control for additional capacity."""
    def __init__(self, channels=96, hidden=34):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GroupNorm(math.gcd(hidden, 8), hidden), nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(hidden, 8), hidden), nn.SiLU(),
            nn.Conv3d(hidden, channels, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, value):
        return value + self.body(value)


def coarse_face_openings(fluid, side):
    """Initial-geometry face proxies; these are not exact pore connectivity.

    Returns B,X,Y,Z,6, ordered -X,+X,-Y,+Y,-Z,+Z. Each coarse face
    averages the minimum fluid fraction across its two adjacent fine planes.
    """
    batch, size, sy, sz = fluid.shape
    if size != sy or size != sz or size % side:
        raise ValueError("Expected a divisible cubic geometry grid")
    step = size // side
    left = torch.arange(step - 1, size - 1, step, device=fluid.device)
    right = left + 1
    openings = fluid.new_zeros(batch, side, side, side, 6)
    for axis in range(3):
        plane = torch.minimum(fluid.index_select(axis + 1, left), fluid.index_select(axis + 1, right))
        plane = plane.movedim(axis + 1, 1)
        plane = plane.reshape(batch, side - 1, side, step, side, step).mean((3, 5))
        face = plane.movedim(1, axis + 1)
        lower = [slice(None)] * 5
        upper = [slice(None)] * 5
        lower[axis + 1] = slice(None, -1)
        upper[axis + 1] = slice(1, None)
        lower[-1] = 2 * axis + 1
        upper[-1] = 2 * axis
        openings[tuple(lower)] = face
        openings[tuple(upper)] = face
    return openings


class LatentGraphBlock(nn.Module):
    """One independent 6-neighbor latent graph per structure, with edge features."""
    def __init__(self, channels=96):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.message = nn.Sequential(nn.Linear(channels + 4, channels), nn.SiLU())
        self.update = nn.Sequential(nn.Linear(2 * channels, channels), nn.SiLU(),
                                    nn.Linear(channels, channels))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, value, geometry):
        batch, channels, side, sy, sz = value.shape
        if side != sy or side != sz:
            raise ValueError("Latent graph requires cubic feature grids")
        coordinates = F.adaptive_avg_pool3d(geometry[:, 2:5], side).flatten(2).transpose(1, 2)
        openings = coarse_face_openings(geometry[:, 0], side).reshape(batch, -1, 6)
        ijk = torch.stack(torch.meshgrid(*(torch.arange(side, device=value.device) for _ in range(3)),
                                         indexing="ij"), dim=-1).reshape(-1, 3)
        offsets = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0],
                                [0, 1, 0], [0, 0, -1], [0, 0, 1]], device=value.device)
        neighbors = (ijk[:, None] + offsets[None]).clamp(0, side - 1)
        index = neighbors[..., 0] * side * side + neighbors[..., 1] * side + neighbors[..., 2]
        nodes = value.flatten(2).transpose(1, 2)
        normalized = self.norm(nodes)
        delta = normalized[:, index] - normalized[:, :, None]
        relative = coordinates[:, index] - coordinates[:, :, None]
        messages = self.message(torch.cat((delta, relative, openings[..., None]), dim=-1))
        aggregate = (messages * openings[..., None]).sum(2) / openings.sum(2).clamp_min(1e-6)[..., None]
        nodes = nodes + self.update(torch.cat((normalized, aggregate), dim=-1))
        return nodes.transpose(1, 2).reshape(batch, channels, side, side, side)


class JointCFDUNet(nn.Module):
    """Geometry/LoGIC encoder, four field channels, and two log-scalar outputs.

field_normalized channels: u/U0, v/U0, w/U0, gauge_pressure/P_ref.
    scalar_standardized channels: standardized ln(K/1m2), ln(delta_p/1Pa).
Normalization statistics are fitted separately on the frozen training set.
"""
    def __init__(self, use_controls: bool = True,
                 widths: Sequence[int] = (16, 32, 64, 96), dropout: float = 0.1,
                 bottleneck_kind: str = "none", extra_depth: int = 2):
        super().__init__()
        if len(widths) != 4 or any(v < 1 for v in widths):
            raise ValueError("Expected four positive encoder widths")
        self.use_controls = use_controls
        self.widths = tuple(widths)
        if use_controls:
            self.control_local = nn.Conv3d(13, 8, 1)
            self.control_global = nn.Sequential(nn.Linear(351, 128), nn.SiLU(),
                                                nn.Dropout(dropout), nn.Linear(128, 64), nn.SiLU())
            self.film = nn.ModuleList(nn.Linear(64, 2 * width) for width in widths)
            for layer in self.film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.encoders.append(ResidualBlock(6 + (8 if use_controls else 0), widths[0]))
        for a, b in zip(widths[:-1], widths[1:]):
            self.downsamples.append(nn.Conv3d(a, b, 2, stride=2))
            self.encoders.append(ResidualBlock(b, b))
        self.decoders = nn.ModuleList(ResidualBlock(widths[i + 1] + widths[i], widths[i])
                                      for i in range(2, -1, -1))
        self.field_head = nn.Conv3d(widths[0], 4, 1)
        pooled_size = 2 * widths[-1] + (64 if use_controls else 0)
        self.scalar_head = nn.Sequential(nn.Linear(pooled_size, 64), nn.SiLU(),
                                         nn.Dropout(dropout), nn.Linear(64, 2))
        if bottleneck_kind not in ("none", "graph", "conv"):
            raise ValueError("Unknown bottleneck kind")
        self.bottleneck_kind = bottleneck_kind
        self.extra_blocks = nn.ModuleList()
        for _ in range(extra_depth if bottleneck_kind != "none" else 0):
            block = LatentGraphBlock(widths[-1]) if bottleneck_kind == "graph" else ExtraConvolution(widths[-1])
            self.extra_blocks.append(block)

    def forward(self, geometry, controls=None):
        if geometry.ndim != 5 or geometry.shape[1] != 6 or any(n % 8 for n in geometry.shape[2:]):
            raise ValueError("Expected Bx6xXxYxZ geometry with axes divisible by eight")
        context = None
        value = geometry
        if self.use_controls:
            if controls is None or controls.shape != (geometry.shape[0], 13, 3, 3, 3):
                raise ValueError("Expected Bx13x3x3x3 controls")
            context = self.control_global(controls.flatten(1))
            # grid_sample addresses W,H,D, while this module stores X,Y,Z.
            query = geometry[:, [4, 3, 2]].permute(0, 2, 3, 4, 1)
            local = F.grid_sample(self.control_local(controls), query,
                                  mode="bilinear", padding_mode="border", align_corners=True)
            value = torch.cat((geometry, local), dim=1)
        elif controls is not None:
            raise ValueError("Geometry-only model does not accept a control branch")
        features = []
        for level, block in enumerate(self.encoders):
            if level:
                value = self.downsamples[level - 1](value)
            value = block(value)
            if context is not None:
                gamma, beta = self.film[level](context).chunk(2, dim=1)
                value = value * (1 + gamma[..., None, None, None]) + beta[..., None, None, None]
            features.append(value)
        for block in self.extra_blocks:
            value = block(value, geometry) if self.bottleneck_kind == "graph" else block(value)
        pooled = [value.mean((2, 3, 4)), value.amax((2, 3, 4))]
        if context is not None:
            pooled.append(context)
        scalars = self.scalar_head(torch.cat(pooled, dim=1))
        for block, skip in zip(self.decoders, reversed(features[:-1])):
            value = F.interpolate(value, size=skip.shape[2:], mode="trilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return {"field_normalized": self.field_head(value), "scalar_standardized": scalars}


class HydraulicReadout(nn.Module):
    """Recover directly predicted K/delta-p and their implied hydraulic flux.

K is apparent full-flow-domain permeability at the prescribed inlet speed.
The pressure field is predicted separately: scalar identity does not imply
that coarse-grid field integrals exactly reproduce these scalar outputs.
"""
    def __init__(self, mean: Sequence[float], scale: Sequence[float],
                 mu_pa_s: float = 0.001, length_m: float = 0.0078125):
        super().__init__()
        values = torch.tensor([mean, scale], dtype=torch.float64)
        if values.shape != (2, 2) or not torch.isfinite(values).all() or (values[1] <= 0).any():
            raise ValueError("Two finite train-fitted means and positive scales are required")
        if min(mu_pa_s, length_m) <= 0:
            raise ValueError("Physical constants must be positive")
        self.register_buffer("mean", values[0])
        self.register_buffer("scale", values[1])
        self.register_buffer("constants", torch.tensor([mu_pa_s, length_m], dtype=torch.float64))

    def forward(self, standardized, area_m2):
        logs = standardized.to(torch.float64) * self.scale + self.mean
        area = torch.as_tensor(area_m2, dtype=torch.float64, device=logs.device)
        if area.shape != (logs.shape[0],) or not torch.isfinite(area).all() or (area <= 0).any():
            raise ValueError("One positive initial-geometry reference area is required per structure")
        mu, length = self.constants.unbind()
        log_k = logs[:, 0]  # reference permeability is 1 m^2
        log_dp = logs[:, 1]  # reference pressure is 1 Pa
        log_q = log_k + log_dp + area.log() - mu.log() - length.log()
        log_u = log_q - area.log()
        return {"delta_p_Pa": log_dp.exp(), "Q_consistency_m3_s": log_q.exp(), "U_consistency_m_s": log_u.exp(),
                "ln_K_domain_app_z": log_k, "K_domain_app_z_m2": log_k.exp(),
                "pressure_gradient_z_Pa_m": log_dp.exp() / length}


def masked_sample_mse(prediction, target, valid):
    """Equal structure weighting; invalid NaNs are selected out before subtraction."""
    if prediction.shape != target.shape or valid.shape != (target.shape[0], *target.shape[2:]):
        raise ValueError("Expected matching B,C,X,Y,Z fields and B,X,Y,Z mask")
    mask = valid[:, None].expand_as(target)
    if not (mask.flatten(1).sum(1) > 0).all():
        raise ValueError("Every structure needs valid target points")
    if not torch.isfinite(target[mask]).all() or not torch.isfinite(prediction[mask]).all():
        raise ValueError("Nonfinite values inside the supervised mask")
    clean_target = torch.where(mask, target, torch.zeros_like(target))
    clean_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
    residual = (clean_prediction - clean_target).square().flatten(1).sum(1)
    return (residual / mask.flatten(1).sum(1)).mean()
