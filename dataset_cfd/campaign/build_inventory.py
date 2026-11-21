"""Build a CFD campaign inventory from prepared (generated + 3-matic meshed)
TPMS structures.

Prepared structures come from the local ``prep`` pipeline
(``dataset_cfd/prep``, a self-contained copy of the FEA prepare stage).  They
are used only as CFD geometry sources: no Abaqus/FEA solve and no mechanical
pairing is implied, so every campaign row is written with ``paired_fea: false``.

Example
-------
python campaign/build_inventory.py \
    --prepared-run prep/runs/my_prepared_run \
    --samples s0000 s0001 s0002 \
    --start-id 100000 \
    --output campaign/inventories/campaign_test.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FAMILY_NAMES = ("Gyroid", "Diamond", "Primitive")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_names(case: Path) -> list[str]:
    with np.load(case / "inputs.npz") as z:
        names = [str(x) for x in z["topology_names"]] if "topology_names" in z else list(FAMILY_NAMES)
        weights = np.asarray(z["weights"], dtype=float)
    used = []
    for index, name in enumerate(names):
        if index < weights.shape[-1] and float(weights[..., index].sum()) > 0.5:
            used.append(name)
    return used or [names[int(np.argmax(weights.sum(axis=(0, 1, 2))))]]


def relative_density(case: Path) -> float:
    with np.load(case / "inputs.npz") as z:
        voxel = np.asarray(z["source_voxel"], dtype=bool)
    return float(voxel.mean())


def choose_split(group: str, train: int = 700, valid: int = 850) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 1000
    if bucket < train:
        return "train"
    if bucket < valid:
        return "validation"
    return "test"


def file_entry(path: Path) -> dict:
    return {"path": path.as_posix(), "exists": path.exists(), "bytes": path.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-run", required=True, help="prep run directory (prep/runs/<batch>)")
    ap.add_argument("--samples", nargs="+", required=True,
                    help="prepared sample ids, e.g. s0500 s0501 (or numeric)")
    ap.add_argument("--start-id", type=int, required=True, help="first campaign sample_id")
    ap.add_argument("--output", required=True)
    ap.add_argument("--kind", default="coverage")
    ap.add_argument("--train-permille", type=int, default=700)
    ap.add_argument("--valid-permille", type=int, default=850)
    args = ap.parse_args()

    run = Path(args.prepared_run).resolve()
    samples = []
    for offset, raw in enumerate(args.samples):
        name = raw if raw.startswith("s") else f"s{int(raw):04d}"
        case = run / "samples" / name
        required = [case / "analysis.inp", case / "remeshed.stl",
                    case / "voxel_fea.npz", case / "inputs.npz"]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            raise SystemExit(f"{case} missing {missing}; run prepare first")
        group = sha256(case / "design.json") if (case / "design.json").exists() else sha256(case / "analysis.inp")
        samples.append(dict(
            sample_id=args.start_id + offset,
            source_sample_id=name,
            source_case=str(case),
            split=choose_split(group, args.train_permille, args.valid_permille),
            geometry_group=group,
            topology_family=family_names(case),
            kind=args.kind,
            relative_density=relative_density(case),
            paired_fea=False,
            files={p.name: file_entry(p) for p in required},
        ))

    counts = {"train": 0, "validation": 0, "test": 0}
    for row in samples:
        counts[row["split"]] += 1
    inventory = {
        "date_asia_shanghai": "campaign-generated",
        "status": "CAMPAIGN_INVENTORY_NOT_A_CFD_RELEASE",
        "scope": "Independent CFD-only structures prepared by the local prep pipeline; no FEA pairing.",
        "proposed_scope": "target campaign labels",
        "sources": {},
        "sample_count": len(samples),
        "split_counts": counts,
        "paired_fea_count": 0,
        "samples": samples,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=1), encoding="utf-8")
    print(json.dumps({"inventory": str(out), "samples": len(samples), "splits": counts}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
