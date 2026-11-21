"""Predict from initial geometry using the released LoGIC ensemble weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def weights(folder, entry):
    path = folder / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Model checksum mismatch: {path.name}")
    return torch.load(path, map_location="cpu", weights_only=True)


def predict_fea(sample, folder, entries, output, device):
    sys.path.insert(0, str(ROOT / "research/fea_prediction"))
    from surrogate.data import read_initial_sample
    from surrogate.inference import derived
    from surrogate.models import CurveNet, predict_neural

    config = read(folder / "physical_config.json")
    c, g, v = read_initial_sample(sample, config, 80)
    inputs = dict(controls=c[None], geometry=g[None], voxel=v[None])
    curves = []
    for entry in entries:
        state = weights(folder, entry)
        model = CurveNet(dict(state, rank=state["basis"].shape[0]), "fusion_pod").to(device)
        model.load_state_dict(state)
        curves.append(predict_neural(model, inputs, device, entry["stress_scale"])[0])
    curve = np.mean(curves, axis=0)
    seed_sd = np.std(curves, axis=0, ddof=1) if len(curves) > 1 else [None] * len(curve)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["strain", "stress_MPa", "seed_sd_MPa"])
        writer.writerows(zip(np.linspace(0, 0.05, len(curve)), curve, seed_sd))
    return derived(curve, float(g[0]), config)


def predict_cfd(sample, folder, entries, output, device, grid):
    sys.path.insert(0, str(ROOT / "research/cfd_prediction"))
    from architecture import HydraulicReadout, JointCFDUNet
    from data import initial_inputs

    if not grid:
        grid = read(sample.parent.parent / "query_grids.json")[sample.name]
    if grid["shape"] != [64,64,64]:
        raise ValueError("The CFD model requires a 64 x 64 x 64 query grid")
    geometry, controls, area, _ = initial_inputs(sample, grid["origin_mm"], grid["spacing_mm"], grid["shape"])
    scaler = read(folder / "scaler.json")
    means = np.asarray(scaler["control_mean"], dtype=np.float32)[:,None,None,None]
    scales = np.asarray(scaler["control_scale"], dtype=np.float32)[:,None,None,None]
    geometry_tensor = torch.from_numpy(geometry)[None].to(device)
    controls_tensor = torch.from_numpy((controls - means) / scales)[None].to(device)
    fields, scalars = [], []
    with torch.inference_mode():
        for entry in entries:
            model = JointCFDUNet(use_controls=True, widths=[16,32,64,96], dropout=0.1,
                                bottleneck_kind="none", extra_depth=0).to(device).eval()
            model.load_state_dict(weights(folder, entry))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(geometry_tensor, controls_tensor)
            fields.append(result["field_normalized"].float().cpu())
            scalars.append(result["scalar_standardized"].float().cpu())
    field = torch.stack(fields).mean(0)[0].numpy()
    field[:3] *= scaler["velocity_ref_m_s"]
    field[3] *= scaler["pressure_ref_Pa"]
    readout = HydraulicReadout(scaler["log_mean"], scaler["log_scale"])
    values = readout(torch.stack(scalars).mean(0), [area])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "field.npz", u=field[0], v=field[1], w=field[2], p_gauge=field[3],
                        geometry_fluid=geometry[1] > 0, origin_mm=grid["origin_mm"], spacing_mm=grid["spacing_mm"])
    return {key: float(value.item()) for key,value in values.items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("task",choices=["fea","cfd"])
    cli.add_argument("--sample",required=True,type=Path)
    cli.add_argument("--models",required=True,type=Path,help="models/ directory from the dataset release")
    cli.add_argument("--output",required=True,type=Path)
    cli.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    cli.add_argument("--seed",type=int,help="Optional single member")
    cli.add_argument("--grid",type=Path,help="CFD grid JSON with origin_mm, spacing_mm and shape for a new design")
    args=cli.parse_args()
    if args.output.exists():
        cli.error("Choose a new output directory")
    if args.grid and args.task != "cfd":
        cli.error("--grid applies to CFD predictions")
    entries=read(args.models / "index.json")[args.task]
    if args.seed is not None:
        entries=[row for row in entries if row["seed"] == args.seed]
        if not entries: cli.error("Unknown ensemble seed")
    torch.set_num_threads(2)
    common=(args.sample.resolve(),args.models/args.task,entries,args.output,torch.device(args.device))
    if args.task=="fea":
        values=predict_fea(*common)
    else:
        values=predict_cfd(*common,read(args.grid) if args.grid else None)
    values["seeds"]=[row["seed"] for row in entries]
    (args.output/"metrics.json").write_text(json.dumps(values,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
