"""Recalculate primary paper metrics from archived predictions and simulation labels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(file):
    with file.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read(file):
    return json.loads(file.read_text(encoding="utf-8"))


def check_fea(release, data):
    results = release / "results/fea"
    with np.load(results / "test_curves.npz", allow_pickle=False) as z:
        ids, saved, predictions = z["sample_ids"], z["truth"], z["members"].mean(0)
    assert sorted(ids) == sorted(read(data / "splits.json")["test"]) and len(ids) == 62
    curves = []
    for sid in ids:
        with np.load(data / "samples" / str(sid) / "curve.npz") as label:
            stress = label["stress_MPa"].astype(np.float32)
        stress[0] = 0.0
        curves.append(stress)
    truth = np.stack(curves)
    np.testing.assert_array_equal(saved, truth)
    error = predictions.astype(np.float64) - truth
    truth = truth.astype(np.float64)
    metrics = dict(ensemble_rmse_MPa=np.sqrt((error**2).mean(axis=1)).mean(),
                   ensemble_mae_MPa=np.abs(error).mean(),
                   ensemble_relative_l2=(np.linalg.norm(error,axis=1)/np.linalg.norm(truth,axis=1)).mean(),
                   ensemble_global_r2=1-(error**2).sum()/((truth-truth.mean())**2).sum())
    recorded = next(row for row in rows(results / "summary.csv") if
                    (row["task"], row["split"], row["method"]) == ("main","test","fusion_pod"))
    for key, value in metrics.items():
        np.testing.assert_allclose(value, float(recorded[key]), rtol=2e-6)
    return metrics


def check_cfd(release, data):
    results = release / "results/cfd"
    recorded = {r["metric"]:float(r["ensemble"]) for r in rows(results / "main_metrics.csv")}
    ids = read(data / "splits.json")["test"]
    assert len(ids) == 75 and set(ids) == {int(f.stem) for f in (results / "predictions").glob("*.npz")}
    measures = []
    for sid in ids:
        sample = data / "samples" / str(sid)
        metrics = read(sample / "metrics.json")
        with np.load(sample / "field.npz") as truth, np.load(results / "predictions" / f"{sid}.npz") as pred:
            mask = truth["mask"]
            np.testing.assert_array_equal(mask, pred["target_valid"])
            assert int(pred["sample_id"]) == sid
            velocity = np.stack([truth[k][mask] for k in ("u","v","w")]).astype(np.float64)
            pressure = truth["p"][mask].astype(np.float64) - metrics["pressure_outlet_Pa"]
            field = pred["field"][:,mask].astype(np.float64)
            lk, ldp = float(pred["log_K"]), float(pred["log_delta_p"])
        measures.append([abs(np.exp(lk)/metrics["K_domain_app_z_m2"]-1),
                         abs(np.exp(ldp)/metrics["delta_p_domain_Pa"]-1),
                         np.linalg.norm(field[:3]-velocity)/np.linalg.norm(velocity),
                         np.linalg.norm(field[3]-pressure)/np.linalg.norm(pressure)])
    keys = ("K_mape_percent", "DP_mape_percent", "relative_velocity_l2", "relative_pressure_l2")
    values = dict(zip(keys, np.mean(measures,axis=0)*100))
    for key, value in values.items():
        np.testing.assert_allclose(value, recorded[key], rtol=2e-5, atol=2e-5)
    return values


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--release",type=Path,required=True)
    cli.add_argument("--data",type=Path,required=True)
    args=cli.parse_args()
    print(json.dumps(dict(fea=check_fea(args.release,args.data/"fea"),
                          cfd_percent=check_cfd(args.release,args.data/"cfd")),indent=2))
