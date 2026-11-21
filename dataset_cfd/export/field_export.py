"""Export one solved CFD case to a regular-grid field array (u,v,w,p).

Reads the native-mesh node field from `fluid_solution.mph` through a COMSOL
Data export (7 columns: x,y,z,u,v,w,p), samples it onto a regular grid with a
KD-tree nearest lookup and masks solid regions using the prepared voxel grid.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _parse_nodes_csv(path: Path) -> dict:
    """Parse a COMSOL Data export (comment lines start with %)."""
    data = np.loadtxt(path, delimiter=None, comments="%")
    if data.ndim != 2 or data.shape[1] < 7:
        raise RuntimeError(f"unexpected COMSOL data table shape {data.shape}")
    return {"x": data[:, 0], "y": data[:, 1], "z": data[:, 2],
            "u": data[:, 6], "v": data[:, 7], "w": data[:, 8], "p": data[:, 9]}


def export_case_field(case: Path, output: Path, grid: int = 64,
                      voxel_path: Path | None = None, timeout_s: int = 900,
                      client=None):
    import mph

    case = Path(case).resolve()
    solution = case / "fluid_solution.mph"
    if not solution.exists():
        raise RuntimeError(f"no solution: {solution}")
    report = json.loads((case / "fluid_report.json").read_text(encoding="utf-8"))
    bounds = np.asarray(report["bounds_mm"], dtype=float)
    own_client = client is None
    if own_client:
        client = mph.start(cores=4, version="6.3")
    model = client.load(str(solution))
    j = model.java
    export_dir = Path(tempfile.mkdtemp(prefix="cfd_field_"))
    csv_path = export_dir / "nodes.txt"
    tag = "field_grid_export"
    e = j.result().export()
    if tag in list(map(str, e.tags())):
        e.remove(tag)
    exp = e.create(tag, "Data")
    exp.set("data", "dset1")
    exp.set("expr", ["x", "y", "z", "u", "v", "w", "p"])
    exp.set("filename", str(csv_path))
    exp.run()
    if not csv_path.exists():
        raise RuntimeError("COMSOL field export produced no file")
    nodes = _parse_nodes_csv(csv_path)
    csv_path.unlink(missing_ok=True)
    export_dir.rmdir()

    lo = bounds[0::2]
    hi = bounds[1::2]
    axes = [np.linspace(lo[k], hi[k], grid) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    tree = cKDTree(np.column_stack([nodes["x"], nodes["y"], nodes["z"]]))
    distance, index = tree.query(points, k=1)
    fluid = np.ones(len(points), dtype=bool)
    if voxel_path is not None and Path(voxel_path).exists():
        with np.load(voxel_path) as z:
            voxel = np.asarray(z["voxel"], dtype=bool)
            origin = np.asarray(z["origin_mm"], dtype=float)
            spacing = np.asarray(z["spacing_mm"], dtype=float)
        idx = np.rint((points - origin) / spacing).astype(np.int64)
        valid = np.all((idx >= 0) & (idx < np.asarray(voxel.shape)), axis=1)
        solid = np.zeros(len(points), dtype=bool)
        solid[valid] = voxel[idx[valid, 0], idx[valid, 1], idx[valid, 2]]
        fluid = (~solid) & (distance <= 0.20)
    else:
        fluid = distance <= 0.20

    out = {"mask": fluid.reshape(grid, grid, grid),
           "origin_mm": lo.astype(np.float32),
           "spacing_mm": ((hi - lo) / (grid - 1)).astype(np.float32),
           "grid": np.int32(grid),
           "source_nodes": int(len(nodes["x"])),
           "distance_max_mm": float(distance[fluid].max()) if fluid.any() else None}
    for key in ("u", "v", "w", "p"):
        values = nodes[key][index]
        array = np.full(len(points), np.nan, dtype=np.float32)
        array[fluid] = values[fluid].astype(np.float32)
        out[key] = array.reshape(grid, grid, grid)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    client.remove(model)
    return {"field_file": str(output), "grid": grid, "nodes": int(len(nodes["x"])),
            "fluid_fraction": float(fluid.mean()),
            "u_max": float(np.nanmax(out["u"])), "p_min": float(np.nanmin(out["p"])),
            "p_max": float(np.nanmax(out["p"]))}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--voxel", default=None)
    args = ap.parse_args()
    print(json.dumps(export_case_field(Path(args.case), Path(args.output), args.grid,
                                       Path(args.voxel) if args.voxel else None), indent=1))
