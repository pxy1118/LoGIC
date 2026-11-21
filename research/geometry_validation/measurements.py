"""Geometry-only measurements; no FEA/CFD response labels are read."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial.transform import Rotation


def phase_connectivity(phase, neighbors=6):
    """Report connected material/pore fractions and face-to-face spanning mass."""
    phase = np.asarray(phase, dtype=bool)
    labels, count = label(phase, generate_binary_structure(3, 1 if neighbors == 6 else 3))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    total = int(phase.sum())
    spanning = []
    for axis in range(3):
        left = np.unique(np.take(labels, 0, axis=axis))
        right = np.unique(np.take(labels, -1, axis=axis))
        common = np.intersect1d(left[left > 0], right[right > 0])
        spanning.append(float(sizes[common].sum() / total) if total else None)
    return {"components": int(count), "voxels": total,
            "largest_fraction": float(sizes.max(initial=0) / total) if total else None,
            "spanning_fraction_xyz": spanning, "neighborhood": neighbors}


def voxel_integrity(mask, sensitivity=False):
    result = {"solid_6": phase_connectivity(mask), "pore_6": phase_connectivity(~mask)}
    if sensitivity:
        result["solid_26"] = phase_connectivity(mask, 26)
        result["pore_26"] = phase_connectivity(~mask, 26)
    return result


def interpolation_weights(coordinates, length):
    t = np.asarray(coordinates) * 2 / length
    return np.maximum(1 - np.abs(t[:, None] - np.arange(3)[None]), 0)


def density_regions(mask, controls, coordinates, length=7.5, bins=3):
    """Compare each regional SVF with the exact mean trilinear target in it.

    Regions are fixed nonoverlapping bins. They are repeated measurements within
    a structure, not independent experimental replicates.
    """
    intervals = [np.array_split(np.arange(n), bins) for n in mask.shape]
    weights = [interpolation_weights(x, length) for x in coordinates]
    averaged = [np.array([w[ids].mean(0) for ids in groups]) for w, groups in zip(weights, intervals)]
    targets = np.einsum("ia,jb,kc,abc->ijk", *averaged, controls)
    rows = []
    for i, xs in enumerate(intervals[0]):
        for j, ys in enumerate(intervals[1]):
            for k, zs in enumerate(intervals[2]):
                actual = float(mask[np.ix_(xs, ys, zs)].mean())
                target = float(targets[i, j, k])
                rows.append({"i": i, "j": j, "k": k, "bins": bins,
                             "target_svf": target, "measured_svf": actual,
                             "error_pp": 100 * (actual - target)})
    global_weights = [w.mean(0) for w in weights]
    target_mean = float(np.einsum("a,b,c,abc", *global_weights, controls))
    errors = np.array([row["error_pp"] for row in rows])
    return {"target_volume_mean": target_mean, "sampled_svf": float(mask.mean()),
            "global_error_pp": 100 * (float(mask.mean()) - target_mean),
            "regional_mae_pp": float(np.mean(np.abs(errors))),
            "regional_max_absolute_error_pp": float(np.max(np.abs(errors))),
            "regional_bias_pp": float(errors.mean()), "regions": rows}


def axial_profile(mask, controls, coordinates, axis, length=7.5):
    # One unit-cell-thick sliding slab, complete cross-section; no padded edges.
    count = max(1, int(round(mask.shape[axis] / 3)))
    measured = mask.mean(tuple(k for k in range(3) if k != axis))
    weights = [interpolation_weights(x, length) for x in coordinates]
    kwargs = [w if k == axis else w.mean(0) for k, w in enumerate(weights)]
    expression = ["ia,b,c,abc->i", "a,ib,c,abc->i", "a,b,ic,abc->i"][axis]
    target = np.einsum(expression, *kwargs, controls)
    kernel = np.ones(count) / count
    return {"axis": axis, "window_samples": count,
            "window_mm": float(count * (coordinates[axis][1] - coordinates[axis][0])),
            "coordinate_mm": np.convolve(coordinates[axis], kernel, "valid"),
            "target_svf": np.convolve(target, kernel, "valid"),
            "measured_svf": np.convolve(measured, kernel, "valid")}


def triangle_aspect_ratios(vertices):
    lengths = np.stack([np.linalg.norm(vertices[:, a] - vertices[:, b], axis=1)
                        for a, b in ((0, 1), (1, 2), (2, 0))], axis=1)
    area = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0],
                                   vertices[:, 2] - vertices[:, 0]), axis=1) / 2
    return np.divide(lengths.max(1) * lengths.sum(1), 4 * np.sqrt(3) * area,
                     out=np.full(len(area), np.inf), where=area > 0)


def surface_integrity(mesh, length=7.5):
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    ratios = triangle_aspect_ratios(np.asarray(mesh.points)[triangles])
    finite = ratios[np.isfinite(ratios)]
    boundary = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                          feature_edges=False, manifold_edges=False).n_cells
    nonmanifold = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=True,
                                             feature_edges=False, manifold_edges=False).n_cells
    connected = mesh.connectivity()
    regions = len(np.unique(connected.cell_data["RegionId"]))
    volume = float(abs(mesh.volume)) if not boundary and not nonmanifold else None
    return {"triangles": int(len(triangles)), "points": int(mesh.n_points),
            "bounds_mm": np.asarray(mesh.bounds).tolist(), "boundary_edges": int(boundary),
            "nonmanifold_edges": int(nonmanifold), "surface_components": regions,
            "volume_mm3": volume, "mesh_svf": volume / length**3 if volume is not None else None,
            "aspect_ratio_definition": "longest_edge * perimeter / (4 * sqrt(3) * triangle_area)",
            "aspect_ratio_median": float(np.median(finite)) if len(finite) else None,
            "aspect_ratio_p95": float(np.quantile(finite, .95)) if len(finite) else None,
            "aspect_ratio_below_5_fraction": float(np.mean(ratios < 5)),
            "degenerate_triangles": int(np.sum(~np.isfinite(ratios))),
            "self_intersection_test": "NOT_PERFORMED", "scope": "exported initial surface, before remeshing or component removal"}


def sample_phase_diagnostics(phases, quaternions, bounds, constant_angles=None):
    """Sparse spatial detail supplements the generator's full-node diagnostics."""
    n = phases[0].shape[0]
    indices = np.unique(np.r_[np.linspace(0, n - 1, 13, dtype=int), n // 2])
    ix = np.ix_(indices, indices, indices)
    step = np.array([(b - a) / (n - 1) for a, b in bounds])
    jac = np.empty((len(indices),) * 3 + (3, 3))
    for comp, phase in enumerate(phases):
        for axis in range(3):
            lo, hi = np.maximum(indices - 1, 0), np.minimum(indices + 1, n - 1)
            low_ids, high_ids = [indices] * 3, [indices] * 3
            low_ids[axis], high_ids[axis] = lo, hi
            shape = [1] * 3; shape[axis] = len(indices)
            jac[..., comp, axis] = (phase[np.ix_(*high_ids)] - phase[np.ix_(*low_ids)]) / ((hi - lo) * step[axis]).reshape(shape)
    target = Rotation.from_quat(quaternions[ix].reshape(-1, 4)).as_matrix().reshape(jac.shape)
    actual = np.empty_like(target)
    errors = []
    for axis in range(3):
        a, b = ((1, 2), (2, 0), (0, 1))[axis]
        direction = np.cross(jac[..., a, :], jac[..., b, :])
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-15)
        dot = np.sum(direction * target[..., :, axis], axis=-1)
        direction *= np.where(dot < 0, -1, 1)[..., None]
        actual[..., :, axis] = direction
        errors.append(np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1))))
    error = np.stack(errors, -1)
    affine_max = None
    if constant_angles is not None:
        rotation = Rotation.from_euler("xyz", constant_angles, degrees=True).as_matrix().T
        axes = [np.linspace(a, b, n) for a, b in bounds]
        center = np.mean(bounds, axis=1)
        affine_max = 0.
        for comp, phase in enumerate(phases):
            expected = center[comp] + sum(rotation[comp, k] * (axes[k] - center[k]).reshape(
                tuple(n if j == k else 1 for j in range(3))) for k in range(3))
            affine_max = max(affine_max, float(np.max(np.abs(phase - expected))))
    positions = np.stack(np.meshgrid(*[np.linspace(a, b, n)[indices] for a, b in bounds], indexing="ij"), -1)
    return {"indices": indices, "positions_generator": positions, "target_axes": target, "actual_axes": actual,
            "axis_error_deg": error, "sampled_detJ": np.linalg.det(jac),
            "sampled_singular_values": np.linalg.svd(jac, compute_uv=False)}, {
                "sparse_nodes": int(error.shape[0]**3),
                "sparse_axis_mean_error_deg": error.mean((0, 1, 2)).tolist(),
                "sparse_axis_p95_error_deg": np.quantile(error.reshape(-1, 3), .95, axis=0).tolist(),
                "constant_affine_max_error_generator_units": affine_max,
                "scope": "coordinate-frame axes, not surface normals or stress directions; axis sign is equivalent"}
