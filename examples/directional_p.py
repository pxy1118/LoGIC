"""P lattice with physical local-z axes turning from vertical to tilted.

Run from the repository: python examples/directional_p.py --resolution 81
Outputs are geometry candidates for FEA, not validated force-redirection devices.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation

from orientation_mapping import interpolate_rotation_grid, integrate_orientation_field, mapping_diagnostics
from tpms_hybrid import Primitive, create_hybrid_tpms_solid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=81)
    parser.add_argument("--top-angle", type=float, default=45.)
    parser.add_argument("--solid-fraction", type=float, default=.3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.resolution < 15 or not 0 < args.solid_fraction < 1 or abs(args.top_angle) > 60:
        parser.error("resolution >= 15, 0 < solid-fraction < 1, |top-angle| <= 60 required")
    out = args.output or Path("output") / ("directional_p_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    n = args.resolution
    bounds = ((-4.5, 4.5), (-4.5, 4.5), (-9., 9.))
    grid = np.zeros((3, 3, 3, 3))
    grid[..., 1] = np.linspace(0, args.top_angle, 3)
    q = interpolate_rotation_grid(grid, n)
    inverse_q = q.copy()
    inverse_q[..., :3] *= -1
    xyz = np.stack(np.meshgrid(*[np.linspace(*b, n) for b in bounds], indexing="ij"), -1)
    phases = integrate_orientation_field(q, bounds)
    old = Rotation.from_quat(q.reshape(-1, 4)).inv().apply(xyz.reshape(-1, 3)).reshape(xyz.shape)
    report = {"parameters": vars(args) | {"output": str(out)}, "ranges": bounds,
              "legacy_comparison": "inverse input quaternion gives the same intended physical orientation",
              "legacy_geometry": mapping_diagnostics(tuple(old[..., i] for i in range(3)), q, bounds),
              "integrated_geometry": mapping_diagnostics(phases, q, bounds),
              "mechanical_validation": "NOT_RUN; no claim about actual stress trajectories"}
    all_masks = []
    for name, rotation, mode in (("unrotated", None, "legacy"),
                                  ("legacy_gradient", inverse_q, "legacy"),
                                  ("integrated_gradient", q, "integrated")):
        mesh, porosity, _, mask = create_hybrid_tpms_solid(
            [Primitive], *bounds, weight_volume=np.ones((n, n, n, 1)),
            density_field=np.full((n, n, n), args.solid_fraction),
            rotation_field=rotation, rotation_mode=mode, resolution=n,
            morph_close=False, remove_small=False, smooth=False, return_mask=True,
            max_axis_error_deg=10 if mode == "integrated" else None,
            max_phase_stretch_ratio=1.6 if mode == "integrated" else None,
        )
        mesh.save(out / f"{name}.stl")
        labels, components = label(mask)
        counts = np.bincount(labels.ravel())[1:]
        mesh_regions = mesh.connectivity()["RegionId"]
        report[name] = {"voxel_solid_fraction": 1 - porosity,
                        "mesh_solid_fraction": abs(mesh.volume) / np.prod([b-a for a,b in bounds]),
                        "voxel_components_6_connected": int(components),
                        "largest_voxel_component_fraction": float(counts.max() / mask.sum()),
                        "mesh_connected_regions": int(np.unique(mesh_regions).size),
                        "mesh_open_edges": mesh.n_open_edges, "mesh_cells": mesh.n_cells}
        all_masks.append(mask)
    np.savez_compressed(out / "design.npz", rotation_grid_deg=grid, physical_quaternions=q,
                        ranges=bounds, unrotated_mask=all_masks[0], legacy_mask=all_masks[1],
                        integrated_mask=all_masks[2])
    # Mid-plane sections: black solid; blue target axis; orange realized axis.
    fig, axes = plt.subplots(1, 3, figsize=(11, 7), constrained_layout=True)
    titles = ("Unrotated P", "Legacy coordinate rotation", "Integrated phase coordinates")
    h = [(b-a)/(n-1) for a,b in bounds]
    target = Rotation.from_quat(q.reshape(-1, 4)).apply([0, 0, 1]).reshape((*q.shape[:3], 3))
    step = max(1, n // 9)
    sel = (slice(step, -step, step), n // 2, slice(step, -step, step))
    for i, (ax, mask, title) in enumerate(zip(axes, all_masks, titles)):
        ax.imshow(mask[:, n//2, :].T, origin="lower", extent=(*bounds[0], *bounds[2]), cmap="Greys",
                  vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        if i:
            coords = tuple(old[..., j] for j in range(3)) if i == 1 else phases
            rows = [np.stack(np.gradient(coords[j], *h), -1) for j in (0, 1)]
            direction = np.cross(rows[0], rows[1])
            direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
            for vector, color in ((target, "#0072B2"), (direction, "#D55E00")):
                ax.quiver(xyz[sel][..., 0], xyz[sel][..., 2], vector[sel][..., 0], vector[sel][..., 2],
                          color=color, angles="xy", scale_units="xy", scale=1.2, width=.006, pivot="mid")
        ax.set(title=title, xlabel="x", ylabel="z")
    fig.suptitle("P lattice: bottom 0° → top %.0f°\nBlue: target local z; orange: realized local z. Geometry only." % args.top_angle)
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
