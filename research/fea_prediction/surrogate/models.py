from __future__ import annotations

import numpy as np
import torch
from torch import nn

BRANCHES = {
    'fusion_pod': 'CGV', 'fusion_direct': 'CGV', 'control_pod': 'CG',
    'controls_only_pod': 'C', 'voxel_only_pod': 'V',
    'controls_voxel_pod': 'CV', 'geometry_voxel_pod': 'GV',
    'control_stats_pod': 'S',
}
BASELINES = ('mean', 'pod_krr', 'density_pod_krr')


def control_statistics(controls):
    x = controls.reshape(-1, 27, 13)
    return np.concatenate([x.mean(1), x.std(1)], axis=1)


def fit_preprocessing(train, rank=8):
    """Fit exclusively on train; validation/test arrays are not accepted."""
    result = {}
    for name in ('controls', 'geometry'):
        x = train[name]
        result[name+'_mean'] = x.mean(0).astype('float32')
        sd = x.std(0)
        result[name+'_std'] = np.where(sd > 1e-6, sd, 1).astype('float32')
    stats = control_statistics(train['controls'])
    result['stats_mean'] = stats.mean(0).astype('float32')
    result['stats_std'] = np.where(stats.std(0) > 1e-6, stats.std(0), 1).astype('float32')
    scale = max(float(np.sqrt(np.mean(train['stress']**2))), 1.)
    y = train['stress']/scale
    mean = y.mean(0)
    _, singular, vt = np.linalg.svd(y-mean, full_matrices=False)
    rank = min(rank, len(y)-1, len(singular))
    if rank < 1:
        raise ValueError('Need at least two independent training structures')
    coefficient_scale = singular[:rank]/np.sqrt(max(len(y)-1, 1))
    basis = vt[:rank]*coefficient_scale[:, None]
    basis[:, 0] = 0
    mean[0] = 0
    result.update(stress_scale=scale, curve_mean=mean.astype('float32'), basis=basis.astype('float32'),
                  rank=rank, pod_train_variance_fraction=float(np.sum(singular[:rank]**2)/max(np.sum(singular**2), 1e-20)))
    return result


def tabular(data, prep, method='pod_krr'):
    if method == 'density_pod_krr':
        return (data['geometry'][:, :1]-prep['geometry_mean'][:1])/prep['geometry_std'][:1]
    return np.concatenate([(data[k]-prep[k+'_mean'])/prep[k+'_std']/np.sqrt(data[k].shape[1])
                           for k in ('controls', 'geometry')], axis=1)


class CurveNet(nn.Module):
    """Small multimodal branch and POD or direct fixed-grid decoder."""

    def __init__(self, prep, method):
        super().__init__()
        if method not in BRANCHES:
            raise ValueError('Unknown neural method: ' + method)
        self.method = method
        self.branches = BRANCHES[method]
        for name in ('controls_mean', 'controls_std', 'geometry_mean', 'geometry_std', 'curve_mean', 'basis'):
            self.register_buffer(name, torch.as_tensor(prep[name]).clone())
        if 'C' in self.branches:
            self.control_encoder = nn.Sequential(nn.Linear(len(prep['controls_mean']), 64), nn.SiLU(),
                                             nn.Linear(64, 32), nn.SiLU())
        if 'G' in self.branches:
            self.geometry_encoder = nn.Sequential(nn.Linear(len(prep['geometry_mean']), 32), nn.SiLU(),
                                              nn.Linear(32, 16), nn.SiLU())
        if 'S' in self.branches:
            for name in ('stats_mean', 'stats_std'):
                self.register_buffer(name, torch.as_tensor(prep[name]).clone())
            self.stats_encoder = nn.Sequential(nn.Linear(26, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.use_voxel = 'V' in self.branches
        if self.use_voxel:
            self.voxel_encoder = nn.Sequential(
                nn.Conv3d(1, 8, 3, stride=2, padding=1), nn.GroupNorm(4, 8), nn.SiLU(),
                nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
                nn.Conv3d(16, 24, 3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
                nn.AdaptiveAvgPool3d(2), nn.Flatten(), nn.Linear(192, 32), nn.SiLU())
        width = sum({'C':32, 'G':16, 'V':32, 'S':32}[b] for b in self.branches)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Dropout(.1),
                                  nn.Linear(64, len(prep['curve_mean']) if method=='fusion_direct' else prep['rank']))
        self.register_buffer('origin_mask', torch.cat([torch.zeros(1), torch.ones(len(prep['curve_mean'])-1)]))

    def forward(self, controls, geometry, voxel):
        parts = []
        if 'C' in self.branches:
            parts.append(self.control_encoder((controls-self.controls_mean)/self.controls_std))
        if 'G' in self.branches:
            parts.append(self.geometry_encoder((geometry-self.geometry_mean)/self.geometry_std))
        if 'S' in self.branches:
            field = controls.reshape(-1, 27, 13)
            stats = torch.cat([field.mean(1), field.std(1, correction=0)], dim=1)
            parts.append(self.stats_encoder((stats-self.stats_mean)/self.stats_std))
        if self.use_voxel:
            parts.append(self.voxel_encoder(voxel))
        latent = self.head(torch.cat(parts, dim=1))
        curve = self.curve_mean + (latent if self.method=='fusion_direct' else latent@self.basis)
        # No monotonicity constraint: actual post-peak softening must remain representable.
        return curve*self.origin_mask


def objective(pred, target, early_weight=.2, work_weight=.05):
    point = torch.mean((pred-target)**2)
    early = torch.mean((pred[:, 10:51]-target[:, 10:51])**2)
    # Integral in strain normalized to [0,1]; scale-independent across samples.
    endpoints = (200, 400, 600, 800, 1000)
    work = torch.stack([(torch.trapezoid(pred[:, :i+1]-target[:, :i+1], dx=.001, dim=1)**2).mean()
                        for i in endpoints]).mean()
    return point + early_weight*early + work_weight*work


def predict_neural(model, data, device, scale, batch_size=16):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data['controls']), batch_size):
            args = [None if k=='voxel' and not model.use_voxel else torch.as_tensor(data[k][start:start+batch_size], device=device)
                    for k in ('controls', 'geometry', 'voxel')]
            predictions.append(model(*args).cpu().numpy()*scale)
    return np.concatenate(predictions)


def mean_rmse(truth, pred):
    return float(np.sqrt(np.mean((np.asarray(truth)-np.asarray(pred))**2, axis=1)).mean())
