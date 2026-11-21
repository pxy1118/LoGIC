"""Export a full CFD dataset in the FEA-dataset layout.

Per sample:
  controls.npz, voxel.npz, features.json, sample.json   (input, FEA-compatible)
  field.npz                                             (u,v,w,p on a regular grid + mask)
  metrics.json                                          (K, lnK, delta-p, gradient, Q, U, flags)
  quality.json                                          (physics checks, mesh quality, provenance)

Aggregated:
  labels.npz, manifest.json, dataset_card.md

Example:
  python export/export_full_dataset.py \
      --runs runs/staged_cfd_v2_session_20261108 \
      --output datasets/cfd_v2_full_20261109 --grid 64 --limit 2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from field_export import export_case_field, _parse_nodes_csv  # noqa: F401  (re-exported helper)

FEA_KEYS = ("weights", "density", "rotation_deg", "topology_names", "quaternion_xyzw")


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_run(run: Path):
    labels = pd.read_csv(run / "results" / "labels.csv")
    labels.insert(0, "run", run.name)
    inventory = json.loads((run / "source_inventory.json").read_text(encoding="utf-8"))
    inv = {int(row["sample_id"]): row for row in inventory["samples"]}
    return labels, inv


def write_inputs(sample_dir: Path, source_case: Path, label: pd.Series):
    sample_dir.mkdir(parents=True, exist_ok=True)
    inputs = source_case / "inputs.npz"
    with np.load(inputs) as z:
        arrays = {key: z[key] for key in FEA_KEYS if key in z}
    np.savez_compressed(sample_dir / "controls.npz", **arrays)
    shutil.copy2(source_case / "voxel_fea.npz", sample_dir / "voxel.npz")
    with np.load(sample_dir / "voxel.npz") as z:
        voxel = np.asarray(z["voxel"], dtype=bool)
        origin = np.asarray(z["origin_mm"], dtype=float)
        spacing = np.asarray(z["spacing_mm"], dtype=float)
    features = {
        "relative_density": float(label["relative_density"]),
        "voxel_fraction": float(voxel.mean()),
        "voxel_shape": list(voxel.shape),
        "voxel_origin_mm": origin.tolist(),
        "voxel_spacing_mm": spacing.tolist(),
        "feature_scope": "initial design geometry; CFD prepared source",
    }
    (sample_dir / "features.json").write_text(json.dumps(features, indent=1), encoding="utf-8")
    sample = {
        "sample_id": int(label["sample_id"]),
        "topology_family": [str(label["topology"])],
        "kind": str(label["kind"]),
        "split": str(label["split"]),
        "geometry_group": str(label["geometry_group"]),
        "paired_fea": bool(label["paired_fea"]),
        "run": str(label.get("run", "")),
        "protocol_sha256": str(label["protocol_sha256"]),
        "source_sample_id": str(label["source_sample_id"]),
    }
    (sample_dir / "sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
    return voxel.shape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--grid", type=int, default=64, help="field grid size per axis")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N labels")
    ap.add_argument("--skip-fields", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    (out / "samples").mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    processed = 0
    client = None  # COMSOL is only started when a field actually has to be extracted
    for raw in args.runs:
        if args.limit and processed >= args.limit:
            break
        run = Path(raw).resolve()
        labels, inv = read_run(run)
        for _, label in labels.iterrows():
            if args.limit and processed >= args.limit:
                break
            sid = int(label["sample_id"])
            if sid in seen:
                raise SystemExit(f"duplicate sample_id across runs: {sid}")
            seen.add(sid)
            case = Path(str(label["case"]))
            files = inv[sid]["files"]
            sample_dir = out / "samples" / str(sid)
            # With --skip-existing a complete sample keeps its exported inputs, so
            # the (possibly deleted) FEA source case is never touched again.
            inputs_ready = ((sample_dir / "controls.npz").is_file()
                            and (sample_dir / "voxel.npz").is_file())
            if not (args.skip_existing and inputs_ready):
                source_path = None
                for key in ("analysis.inp", "remeshed.stl", "voxel_fea.npz", "inputs.npz"):
                    if key in files:
                        source_path = Path(files[key]["path"])
                        break
                if source_path is None:
                    raise SystemExit(f"sample {sid}: no source file in inventory")
                write_inputs(sample_dir, source_path.parent, label)
            field_path = sample_dir / "field.npz"
            if not args.skip_fields and not field_path.exists():
                if client is None:
                    import mph
                    client = mph.start(cores=4, version="6.3")
                voxel_path = sample_dir / "voxel.npz"
                info = export_case_field(case, field_path, grid=args.grid, voxel_path=voxel_path, client=client)
            case_results = {}
            if (case / "results.json").exists():
                case_results = json.loads((case / "results.json").read_text(encoding="utf-8"))
            metrics = {
                "sample_id": sid,
                "pressure_inlet_Pa": case_results.get("inlet_average_pressure_Pa"),
                "pressure_outlet_Pa": case_results.get("outlet_average_pressure_Pa"),
                "mesh_min_skewness": (case_results.get("mesh") or {}).get("min_quality"),
                "mesh_min_volcircum_quality": (case_results.get("mesh") or {}).get("min_tet_volcircum_quality"),
                "volume_closure_rel": (case_results.get("mesh") or {}).get("volume_closure_rel"),
                "K_domain_app_z_m2": float(label["K_domain_app_z_m2"]),
                "ln_K_domain_app_z": float(label["ln_K_domain_app_z"]),
                "delta_p_domain_Pa": float(label["delta_p_domain_Pa"]),
                "pressure_gradient_z_Pa_m": float(label["pressure_gradient_z_Pa_m"]),
                "Q_m3_s": float(label["Q_m3_s"]),
                "U_superficial_m_s": float(label["Q_m3_s"]) / 5.531194384e-5,
                "mass_imbalance_rel": float(label["mass_imbalance_rel"]),
                "tetrahedra": int(label["tetrahedra"]),
                "solve_elapsed_s": float(label["solve_elapsed_s"]),
                "Re_sqrtK": 1000.0 * (float(label["Q_m3_s"]) / 5.531194384e-5) * float(np.sqrt(label["K_domain_app_z_m2"])) / 0.001,
                "split": str(label["split"]),
                "topology": str(label["topology"]),
                "kind": str(label["kind"]),
                "protocol_sha256": str(label["protocol_sha256"]),
                "production_valid": False,
                "pressure_definition": "area-averaged pressure at outer buffer-end sections; delta_p = p_in - p_out",
                "K_definition": "K = mu*(Q/A_ref)*L_meas/delta_p, L_meas=0.0078125 m, A_ref=5.531194e-5 m^2",
                "field_grid": args.grid,
            }
            (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
            physics = {}
            if (case / "physics_verification.json").exists():
                physics = json.loads((case / "physics_verification.json").read_text(encoding="utf-8"))
            quality = {"production_valid": False,
                       "cfd_valid": bool((case_results or {}).get("cfd_valid", True)),
                       "physics_passed": physics.get("passed"),
                       "physics_checks": physics.get("checks"),
                       "mesh_quality_review_required": (case_results.get("mesh") or {}).get("quality_review_required"),
                       "note": "development CFD label; physics checks and mesh flags copied from the solved case receipts"}
            (sample_dir / "quality.json").write_text(json.dumps(quality, indent=1), encoding="utf-8")
            rows.append(label.to_dict())
            processed += 1

    labels_table = pd.DataFrame(rows) if rows else pd.DataFrame()
    labels_table.to_csv(out / "labels.csv", index=False)
    splits: dict[str, list[int]] = {}
    if len(labels_table):
        for split, group in labels_table.groupby("split"):
            splits[str(split)] = sorted(int(x) for x in group["sample_id"])
    (out / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")

    ids = sorted(int(r["sample_id"]) for r in rows) if rows else []
    arrays = {
        "sample_ids": np.asarray(ids, dtype=np.int64),
        "split": np.asarray([str(r["split"]) for r in rows]) if rows else np.asarray([]),
        "K_domain_app_z_m2": np.asarray([float(r["K_domain_app_z_m2"]) for r in rows]) if rows else np.asarray([]),
        "ln_K_domain_app_z": np.asarray([float(r["ln_K_domain_app_z"]) for r in rows]) if rows else np.asarray([]),
        "delta_p_domain_Pa": np.asarray([float(r["delta_p_domain_Pa"]) for r in rows]) if rows else np.asarray([]),
        "pressure_gradient_z_Pa_m": np.asarray([float(r["pressure_gradient_z_Pa_m"]) for r in rows]) if rows else np.asarray([]),
        "Q_m3_s": np.asarray([float(r["Q_m3_s"]) for r in rows]) if rows else np.asarray([]),
    }
    np.savez_compressed(out / "labels.npz", **arrays)
    summary = {"runs": [Path(r).name for r in args.runs],
               "labels": int(len(rows)),
               "splits": {k: len(v) for k, v in splits.items()},
               "field_grid": None if args.skip_fields else [args.grid] * 3,
               "package": str(out)}
    (out / "manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "dataset_card.md").write_text(
        "# CFD full dataset (inputs + fields + scalars)\n\n"
        f"- runs: {', '.join(summary['runs'])}\n"
        f"- labels: {summary['labels']} (splits: {summary['splits']})\n"
        f"- field grid: {('skipped' if args.skip_fields else f'{args.grid}^3')}\n\n"
        "Per sample: `controls.npz`, `voxel.npz`, `features.json`, `sample.json` mirror the FEA dataset layout; "
        "`field.npz` stores u,v,w,p (m/s, Pa) on a regular grid with a fluid mask; `metrics.json` stores K, "
        "lnK, delta-p, pressure gradient, Q and U. Labels are development CFD labels (`production_valid=0`).\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
