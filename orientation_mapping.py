"""Compatible TPMS phase coordinates for prescribed physical lattice orientations.

Q maps lattice axes into world axes; grad(phase) therefore targets Q.T.
Spatial rotations are generally incompatible: integration is a least-squares
projection, NOT an exact realization of an arbitrary rotation field.
"""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import zoom, gaussian_filter
from scipy.spatial.transform import Rotation


def interpolate_rotation_grid(euler_grid, resolution, smooth_sigma=None, method="linear"):
    """Interpolate an arbitrary rectangular xyz Euler-angle grid (degrees).

Unwrap quaternion signs on a spanning tree before component interpolation.
This handles ordinary angle wrapping, not defects/disclinations in an SO(3) field.
"""
    grid = np.asarray(euler_grid, dtype=float)
    if grid.ndim != 4 or grid.shape[-1] != 3 or min(grid.shape[:3]) < 2:
        raise ValueError("Euler grid must have shape (nx, ny, nz, 3), each axis >= 2")
    if not np.isfinite(grid).all() or resolution < 3:
        raise ValueError("Finite angles and resolution >= 3 required")
    if method not in ("linear", "nearest"):
        raise ValueError("method must be linear or nearest")
    q = Rotation.from_euler("xyz", grid.reshape(-1, 3), degrees=True).as_quat()
    q = q.reshape((*grid.shape[:3], 4))
    for idx in np.ndindex(grid.shape[:3]):
        parent = list(idx)
        for axis in (2, 1, 0):
            if parent[axis]:
                parent[axis] -= 1
                if np.dot(q[idx], q[tuple(parent)]) < 0:
                    q[idx] *= -1
                break
    # A remaining negative edge indicates an inconsistent lift or a variation
    # too large for this interpolation. Do not silently interpolate across it.
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = lo.copy()
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        if np.any(np.sum(q[tuple(lo)] * q[tuple(hi)], axis=-1) < -1e-7):
            raise ValueError("Rotation grid has incompatible quaternion signs; refine/simplify the field")
    q = zoom(q, tuple(resolution / n for n in grid.shape[:3]) + (1,),
             order=1 if method == "linear" else 0)
    if smooth_sigma:
        for component in range(4):
            q[..., component] = gaussian_filter(q[..., component], smooth_sigma)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("Degenerate interpolated quaternion")
    return (q / norm).astype(np.float32)


def _validated_field(quaternions, shape):
    q = np.asarray(quaternions, dtype=float)
    if q.shape != (*shape, 4) or not np.isfinite(q).all():
        raise ValueError("rotation_field must be finite with shape (*grid_shape, 4), xyzw")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("Zero quaternion in rotation_field")
    return q / norm


def integrate_orientation_field(quaternions, ranges):
    """Minimize sum_edges |D_j phase_i - average_edge(Q.T_ij)|^2.

Free (natural) boundaries, physical grid spacing, and a mean-coordinate gauge.
The Cartesian graph Laplacian is solved directly by a Neumann DCT; no periodic
boundary assumption. Constant orientation yields the exact affine coordinates.
    """
    shape = np.shape(quaternions)[:3]
    if len(shape) != 3 or min(shape) < 3:
        raise ValueError("At least 3 samples per spatial axis required")
    bounds = np.asarray(ranges, dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(np.diff(bounds) <= 0):
        raise ValueError("ranges must contain three finite increasing pairs")
    q = _validated_field(quaternions, shape)
    h = (bounds[:, 1] - bounds[:, 0]) / (np.array(shape) - 1)
    eigenvalues = np.zeros(shape)
    for axis, n in enumerate(shape):
        view = [1, 1, 1]
        view[axis] = n
        eigenvalues += (4 * np.sin(np.pi * np.arange(n) / (2 * n)) ** 2
                        / h[axis] ** 2).reshape(view)
    eigenvalues[0, 0, 0] = 1  # constant mode fixed separately
    phases = []
    for component in range(3):
        # Q column = Q.T row; avoid retaining a full volume of 3x3 matrices.
        axis_vector = np.eye(3)[component]
        row = Rotation.from_quat(q.reshape(-1, 4)).apply(axis_vector).reshape((*shape, 3))
        rhs = np.zeros(shape)
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = lo.copy()
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            edge = (row[lo + (axis,)] + row[hi + (axis,)]) / (2 * h[axis])
            rhs[lo] -= edge
            rhs[hi] += edge
        spectrum = dctn(rhs, type=2, norm="ortho") / eigenvalues
        spectrum[0, 0, 0] = 0
        phase = idctn(spectrum, type=2, norm="ortho") + bounds[component].mean()
        phases.append(phase)
    return tuple(phases)


def mapping_diagnostics(phases, quaternions, ranges):
    """Sampled Jacobian quality and physical local-z alignment (not FEA).

Actual coordinate-line direction is inv(J) e_z. It differs from both
the surface normal and the polar-rotation axis when the map is strained.
Statistics cover all grid nodes; they cannot certify continuum injectivity.
    """
    shape = phases[0].shape
    q = _validated_field(quaternions, shape)
    h = [(b - a) / (n - 1) for (a, b), n in zip(ranges, shape)]
    determinant_min, stretch_min, stretch_max = np.inf, np.inf, 0.0
    angle_sum, angle_max, residual_sum, count, nonpositive = 0., 0., 0., 0, 0
    for x in range(shape[0]):
        start, stop = max(0, x - 1), min(shape[0], x + 2)
        j = np.empty((*shape[1:], 3, 3))
        for component, phase in enumerate(phases):
            j[..., component, 0] = ((phase[min(x + 1, shape[0] - 1)] - phase[max(x - 1, 0)])
                                    / ((stop - start - 1) * h[0]))
            j[..., component, 1] = np.gradient(phase[x], h[1], axis=0)
            j[..., component, 2] = np.gradient(phase[x], h[2], axis=1)
        det = np.linalg.det(j)
        sv = np.linalg.svd(j, compute_uv=False)
        determinant_min = min(determinant_min, float(det.min()))
        stretch_min = min(stretch_min, float(sv.min()))
        stretch_max = max(stretch_max, float(sv.max()))
        nonpositive += int(np.count_nonzero(det <= 0))
        target = Rotation.from_quat(q[x].reshape(-1, 4)).as_matrix().reshape((*shape[1:], 3, 3))
        residual_sum += float(np.sum((j - target.swapaxes(-1, -2)) ** 2))
        # Cross-product is proportional to inv(J)e_z; abs(dot) treats an axis
        # as unoriented, appropriate for the cubic P lattice.
        actual = np.cross(j[..., 0, :], j[..., 1, :])
        norm = np.linalg.norm(actual, axis=-1)
        cosine = np.abs(np.sum(actual * target[..., :, 2], axis=-1)) / np.maximum(norm, 1e-15)
        angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
        angle_sum += float(angle.sum())
        angle_max = max(angle_max, float(angle.max()))
        count += angle.size
    return {"min_determinant": determinant_min, "nonpositive_jacobian_nodes": nonpositive,
            "min_phase_stretch": stretch_min, "max_phase_stretch": stretch_max,
            "local_z_error_mean_deg": angle_sum / count, "local_z_error_max_deg": angle_max,
            "gradient_rms_error": float(np.sqrt(residual_sum / (3 * count))),
            "scope": "sampled geometry only; not stress direction or mechanical validation"}
