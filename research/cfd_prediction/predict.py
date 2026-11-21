"""Inference reads initial controls/voxel/grid only; no metrics or field target."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from architecture import HydraulicReadout
from data import initial_inputs, read, save, digest
from experiment import initialize_model, resolve_device
from formal import completed_training, verify_identity


def predict_initial(root, sample, output, grid_json=None, seed=None):
    verify_identity(root)
    config = read(root / "config.json")
    checkpoints = completed_training(root, config)
    if seed is not None:
        checkpoints = [item for item in checkpoints if item["seed"] == seed]
        if not checkpoints:
            raise ValueError("Requested seed is not in the frozen trained models")
    if grid_json:
        grid = read(grid_json)
        origin, spacing, shape = grid["origin_mm"], grid["spacing_mm"], grid["shape"]
    else:
        origin = [.0314, .0314, -.15625]
        shape = [64, 64, 64]
        spacing = (np.array([7.4372, 7.4372, 7.8125]) / 63).tolist()
    if shape != [64, 64, 64]:
        raise ValueError("The selected model uses a 64-cubed query grid")
    geometry, controls, area, input_seconds = initial_inputs(sample, origin, spacing, shape)
    normalization = read(root / "snapshot/cache/manifest.json")["scaler"]
    controls = (controls - np.array(normalization["control_mean"], dtype=np.float32)[:, None, None, None]) / np.array(normalization["control_scale"], dtype=np.float32)[:, None, None, None]
    device = resolve_device(config)
    geometry_tensor = torch.from_numpy(geometry)[None].to(device)
    controls_tensor = torch.from_numpy(controls)[None].to(device)
    field_results, scalar_results = [], []
    with torch.inference_mode():
        for item in checkpoints:
            model = initialize_model(config, "none", item["seed"]).eval()
            model.load_state_dict(torch.load(item["checkpoint"], map_location="cpu", weights_only=False)["model"])
            with torch.autocast(device.type, dtype=torch.float16, enabled=config["training"]["amp"] and device.type == "cuda"):
                result = model(geometry_tensor, controls_tensor)
            field_results.append(result["field_normalized"].float().cpu())
            scalar_results.append(result["scalar_standardized"].float().cpu())
            del model
    field = torch.stack(field_results).mean(0)[0].numpy()
    field[:3] *= .001; field[3] *= normalization["pressure_ref_Pa"]
    scalar = torch.stack(scalar_results).mean(0)
    readout = HydraulicReadout(normalization["log_mean"], normalization["log_scale"])
    metrics = {key: float(value[0]) for key, value in readout(scalar, [area]).items()}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "prediction.npz", field=field, u=field[0], v=field[1], w=field[2], p_gauge=field[3],
                        geometry_fluid=geometry[1] > 0, origin_mm=np.asarray(origin), spacing_mm=np.asarray(spacing),
                        units=np.array(["m/s", "m/s", "m/s", "Pa"]))
    save(output / "prediction.json", {"metrics": metrics, "seeds": [item["seed"] for item in checkpoints],
                                      "input_construction_seconds": input_seconds,
                                      "input_files": ["controls.npz", "voxel.npz"] + ([str(grid_json)] if grid_json else []),
                                      "field_resolution": shape, "pressure_scope": "outlet-referenced gauge pressure",
                                      "K_scope": "apparent full-flow-domain z permeability at prescribed 1mm/s",
                                      "source_manifest_sha256": digest(root / "source_manifest.json")})
    return metrics


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True); parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--grid-json", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(predict_initial(args.study, args.inputs, args.output, args.grid_json, args.seed), indent=2))


if __name__ == "__main__":
    main()
