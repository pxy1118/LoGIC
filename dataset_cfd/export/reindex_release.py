"""Build a clean release package with contiguous sample ids from a campaign
dataset whose ids come from several production batches.

Output:
  samples/0 ... samples/<N-1>
  labels.csv / labels.npz / splits.json / manifest.json / id_mapping.csv
  figures/ (test section images only)
  dataset_card.md
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    src = Path(args.source).resolve()
    dst = Path(args.output).resolve()
    if dst.exists():
        raise SystemExit(f"output already exists: {dst}")
    labels = pd.read_csv(src / "labels.csv").sort_values("sample_id").reset_index(drop=True)
    if labels["sample_id"].duplicated().any():
        raise SystemExit("duplicate source sample ids")
    (dst / "samples").mkdir(parents=True)
    (dst / "figures").mkdir()
    mapping = []
    new_labels = []
    splits: dict[str, list[int]] = {}
    arrays = {"sample_ids": [], "split": [], "K_domain_app_z_m2": [],
              "ln_K_domain_app_z": [], "delta_p_domain_Pa": [],
              "pressure_gradient_z_Pa_m": [], "Q_m3_s": []}
    for new_id, row in labels.iterrows():
        old_id = int(row["sample_id"])
        new_id = int(new_id)
        src_dir = src / "samples" / str(old_id)
        if not src_dir.is_dir():
            raise SystemExit(f"missing sample dir: {src_dir}")
        dst_dir = dst / "samples" / str(new_id)
        shutil.copytree(src_dir, dst_dir)
        sample = json.loads((dst_dir / "sample.json").read_text(encoding="utf-8"))
        sample["sample_id"] = new_id
        sample["original_sample_id"] = old_id
        (dst_dir / "sample.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")

        new_row = dict(row)
        new_row["sample_id"] = new_id
        new_row["original_sample_id"] = old_id
        new_labels.append(new_row)
        mapping.append({"new_sample_id": new_id, "original_sample_id": old_id,
                        "run": row.get("run", ""), "split": row["split"],
                        "topology": row["topology"],
                        "protocol_sha256": row["protocol_sha256"]})
        splits.setdefault(str(row["split"]), []).append(new_id)
        arrays["sample_ids"].append(new_id)
        arrays["split"].append(str(row["split"]))
        arrays["K_domain_app_z_m2"].append(float(row["K_domain_app_z_m2"]))
        arrays["ln_K_domain_app_z"].append(float(row["ln_K_domain_app_z"]))
        arrays["delta_p_domain_Pa"].append(float(row["delta_p_domain_Pa"]))
        arrays["pressure_gradient_z_Pa_m"].append(float(row["pressure_gradient_z_Pa_m"]))
        arrays["Q_m3_s"].append(float(row["Q_m3_s"]))

        figure = src / "figures" / f"sample_{old_id:06d}_sections.png"
        if figure.exists():
            shutil.copy2(figure, dst / "figures" / f"sample_{new_id:06d}_sections.png")

    table = pd.DataFrame(new_labels)
    table.to_csv(dst / "labels.csv", index=False)
    pd.DataFrame(mapping).to_csv(dst / "id_mapping.csv", index=False)
    np.savez_compressed(dst / "labels.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    (dst / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")

    protocols = table["protocol_sha256"].value_counts().to_dict()
    summary = {
        "source_dataset": str(src),
        "labels": int(len(table)),
        "splits": {k: len(v) for k, v in splits.items()},
        "sample_id_range": [0, int(len(table)) - 1],
        "field_grid": [64, 64, 64],
        "protocols": protocols,
        "production_valid": int(table["production_valid"].sum()),
        "note": "Contiguous release ids; original ids and run provenance are kept in id_mapping.csv / sample.json.",
    }
    (dst / "manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    card = f"""# CFD dataset (contiguous ids)

- source dataset: `{src.name}`
- samples: **{summary['labels']}** (train {len(splits.get('train', []))} / validation {len(splits.get('validation', []))} / test {len(splits.get('test', []))})
- sample ids: **0 – {summary['labels'] - 1}**; original ids/runs are in `id_mapping.csv`
- field grid: 64³ (`field.npz`: u, v, w, p + mask)
- inputs: `controls.npz`, `voxel.npz` (240³), `features.json`, `sample.json`
- labels: `metrics.json` / `quality.json` per sample, `labels.csv` / `labels.npz` aggregate
- section figures: test split only (`figures/`)
- `production_valid = 0` (development labels)

## Protocol note

{len(protocols)} protocol hashes are present: the early batch used the inp-extracted surface, the later
batch uses `remeshed.stl` directly. A/B validation showed both surfaces are point-identical; K differs by
about 0.1% due to 3-matic remesh non-determinism. The `protocol_sha256` column distinguishes them.

## Regeneration

MPH files are not kept. Each sample can be regenerated from the prepared FEA structure's
`remeshed.stl` + design inputs; the extraction recipe is frozen in the campaign scripts.
"""
    (dst / "dataset_card.md").write_text(card, encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
