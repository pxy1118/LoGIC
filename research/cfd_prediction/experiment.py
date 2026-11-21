"""Internal validation-only selection, followed by a gated selected-model study."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from architecture import JointCFDUNet
from data import CachedDataset, digest, freeze_snapshot, prepare_cache, read, save


class StopRequested(Exception):
    pass


@contextmanager
def study_lock(root):
    import msvcrt
    with (root / ".study.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another controller owns this study") from error
        try:
            yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def atomic_checkpoint(path, data):
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary); os.replace(temporary, path)


def resolve_device(config):
    requested = config["training"].get("device", "cuda")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select cpu explicitly or fix the runtime")
    return torch.device(requested)


def scalar_metrics(rows):
    result = {}
    for key in ("relative_velocity_l2", "relative_pressure_l2", "velocity_rmse_m_s", "pressure_rmse_Pa",
                "lnK_mae", "lnDP_mae", "K_mape_percent", "DP_mape_percent", "flux_consistency_relative_error"):
        result[key] = float(np.mean([row[key] for row in rows]))
    for prefix in ("K", "DP"):
        truth = np.array([row[f"{prefix}_true"] for row in rows])
        predicted = np.array([row[f"{prefix}_pred"] for row in rows])
        result[f"{prefix}_mae"] = float(np.mean(np.abs(truth - predicted)))
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        result[f"{prefix}_r2"] = float(1 - np.sum((predicted - truth) ** 2) / denominator) if denominator > 0 else None
    result["score"] = result["relative_velocity_l2"] + result["relative_pressure_l2"] + .5 * (result["lnK_mae"] + result["lnDP_mae"])
    result["structures"] = len(rows)
    return result


def make_scales(meta, device):
    scales = meta["scaler"]
    return {"control_mean": torch.tensor(scales["control_mean"], device=device)[None, :, None, None, None],
            "control_scale": torch.tensor(scales["control_scale"], device=device)[None, :, None, None, None],
            "log_mean": torch.tensor(scales["log_mean"], device=device),
            "log_scale": torch.tensor(scales["log_scale"], device=device),
            "field_scale": torch.tensor([.001, .001, .001, scales["pressure_ref_Pa"]], device=device)[None, :, None, None, None]}


def batch_inputs(batch, scales, device):
    geometry = batch["geometry"].to(device, non_blocking=True)
    controls = batch["controls"].to(device, non_blocking=True)
    controls = (controls - scales["control_mean"]) / scales["control_scale"]
    target = batch["field"].to(device, non_blocking=True) / scales["field_scale"]
    logs = batch["logs"].to(device, dtype=torch.float32, non_blocking=True)
    physics = batch["physics"].to(device, dtype=torch.float32, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True).bool()
    return geometry, controls, target, logs, physics, valid


def losses(result, target, logs, physics, valid, scales, config):
    predicted = result["field_normalized"].float()
    residual = (predicted - target).square() * valid[:, None]
    count = valid.sum((1, 2, 3)).clamp_min(1)
    velocity = (residual[:, :3].sum((1, 2, 3, 4)) / (3 * count)).mean()
    pressure = (residual[:, 3].sum((1, 2, 3)) / count).mean()
    normalized_log = result["scalar_standardized"].float()
    scalar = (normalized_log - (logs - scales["log_mean"]) / scales["log_scale"]).square().mean()
    recovered = normalized_log * scales["log_scale"] + scales["log_mean"]
    log_flux = recovered.sum(1) + physics[:, 1].log() - math.log(.001 * .0078125)
    consistency = (log_flux - physics[:, 0].log()).square().mean()
    weights = config["loss"]
    total = weights["velocity"] * velocity + weights["pressure"] * pressure + weights["log_scalars"] * scalar + weights["flux_consistency"] * consistency
    return total


@torch.inference_mode()
def evaluate(models, snapshot, split, config, output=None, state_callback=None, baseline=None):
    device = resolve_device(config)
    dataset = CachedDataset(snapshot, split)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=True)
    scales = make_scales(dataset.meta, device)
    for model in models:
        model.eval()
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in loader:
        geometry, controls, target, logs, physics, valid = batch_inputs(batch, scales, device)
        if baseline is None:
            results = []
            with torch.autocast(device.type, dtype=torch.float16, enabled=config["training"]["amp"] and device.type == "cuda"):
                for model in models:
                    results.append(model(geometry, controls))
            predicted_field = torch.stack([result["field_normalized"].float() for result in results]).mean(0)
            predicted_logs = torch.stack([result["scalar_standardized"].float() for result in results]).mean(0) * scales["log_scale"] + scales["log_mean"]
        else:
            predicted_field = baseline[0].to(device)[None].expand(len(geometry), -1, -1, -1, -1)
            predicted_logs = scales["log_mean"][None].expand(len(geometry), -1)
        if not torch.isfinite(predicted_field).all() or not torch.isfinite(predicted_logs).all():
            raise RuntimeError(f"Nonfinite evaluation outputs for structures {batch['sample_id'].tolist()}; checkpoints retained")
        error = (predicted_field - target).square() * valid[:, None]
        target_square = target.square() * valid[:, None]
        velocity_error = error[:, :3].sum((1, 2, 3, 4))
        pressure_error = error[:, 3].sum((1, 2, 3))
        relative_u = (velocity_error / target_square[:, :3].sum((1, 2, 3, 4)).clamp_min(1e-12)).sqrt().cpu().numpy()
        relative_p = (pressure_error / target_square[:, 3].sum((1, 2, 3)).clamp_min(1e-12)).sqrt().cpu().numpy()
        count = valid.sum((1, 2, 3)).clamp_min(1)
        rmse_u = (velocity_error / (3 * count)).sqrt().cpu().numpy() * .001
        rmse_p = (pressure_error / count).sqrt().cpu().numpy() * dataset.meta["scaler"]["pressure_ref_Pa"]
        reference_logs, pred_logs = batch["logs"].numpy().astype(float), predicted_logs.cpu().numpy().astype(float)
        reference_physics = batch["physics"].numpy().astype(float)
        for i, sid in enumerate(batch["sample_id"].tolist()):
            lk, ld = pred_logs[i]
            reference_k, reference_dp = np.exp(reference_logs[i])
            k, dp = np.exp([lk, ld])
            q = float(np.exp(lk + ld) * reference_physics[i, 1] / (.001 * .0078125))
            if not np.isfinite([k, dp, q]).all():
                raise RuntimeError(f"Scalar conversion overflow for structure {sid}; checkpoints retained")
            row = {"sample_id": sid, "relative_velocity_l2": float(relative_u[i]), "relative_pressure_l2": float(relative_p[i]),
                   "velocity_rmse_m_s": float(rmse_u[i]), "pressure_rmse_Pa": float(rmse_p[i]),
                   "lnK_mae": float(abs(lk - reference_logs[i, 0])), "lnDP_mae": float(abs(ld - reference_logs[i, 1])),
                   "K_mape_percent": float(abs(k / reference_k - 1) * 100), "DP_mape_percent": float(abs(dp / reference_dp - 1) * 100),
                   "K_true": float(reference_k), "K_pred": float(k), "DP_true": float(reference_dp), "DP_pred": float(dp),
                   "Q_true": float(reference_physics[i, 0]), "Q_consistency": q,
                   "flux_consistency_relative_error": float(abs(q / reference_physics[i, 0] - 1)),
                   "valid_grid_points": int(count[i].item())}
            rows.append(row)
            if output:
                physical_field = (predicted_field[i] * scales["field_scale"][0]).cpu().numpy()
                with np.load(Path(snapshot) / "samples" / str(sid) / "field.npz") as grid:
                    origin, spacing = grid["origin_mm"], grid["spacing_mm"]
                path = Path(output) / f"{sid}.npz"
                temporary = path.with_name(path.name + ".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, field=physical_field, u=physical_field[0], v=physical_field[1],
                                        w=physical_field[2], p_gauge=physical_field[3],
                                        target_valid=valid[i].cpu().numpy(), geometry_fluid=(geometry[i, 1] > 0).cpu().numpy(),
                                        origin_mm=origin, spacing_mm=spacing, field_units=np.array(["m/s", "m/s", "m/s", "Pa"]),
                                        log_K=lk, log_delta_p=ld, sample_id=sid)
                os.replace(temporary, path)
        if state_callback:
            state_callback({"evaluated": len(rows), "evaluation_total": len(dataset)})
    return scalar_metrics(rows), rows


def initialize_model(config, variant, seed):
    device = resolve_device(config)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return JointCFDUNet(use_controls=True, widths=config["model"]["widths"],
                         dropout=config["model"]["dropout"], bottleneck_kind=variant,
                         extra_depth=config["model"]["extra_depth"]).to(device)


def train_run(root, snapshot, config, variant, seed, epochs, patience, stage, state, stop_check=None):
    directory = root / "runs" / f"{stage}_{variant}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": digest(root / "config.json"), "cache_sha256": digest(snapshot / "cache/manifest.json"),
                "sources_sha256": digest(root / "source_manifest.json"), "variant": variant, "seed": seed,
                "epochs": epochs, "patience": patience, "stage": stage}
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
        if result["identity"] != identity:
            raise RuntimeError("Completed run identity changed; use a new study")
        return result
    dataset = CachedDataset(snapshot, "train")
    device = resolve_device(config)
    scales = make_scales(dataset.meta, device)
    model = initialize_model(config, variant, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] and device.type == "cuda")
    epoch_start, history, best_score, best_epoch, stale, elapsed = 0, [], float("inf"), -1, 0, 0.
    last_path = directory / "last.pt"
    if last_path.exists():
        previous = torch.load(last_path, map_location="cpu", weights_only=False)
        if previous["identity"] != identity:
            raise RuntimeError("Resume code/config/data mismatch; old checkpoint retained")
        model.load_state_dict(previous["model"]); optimizer.load_state_dict(previous["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != "step":
                    values[key] = value.to(device)
        scaler.load_state_dict(previous["grad_scaler"])
        epoch_start = previous["epoch"] + 1
        history, best_score, best_epoch = previous["history"], previous["best_score"], previous["best_epoch"]
        stale, elapsed = previous["stale"], previous["elapsed_s"]
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        random.setstate(previous["python_rng"]); np.random.set_state(previous["numpy_rng"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    remaining_epochs = () if stale >= patience else range(epoch_start, epochs)
    for epoch in remaining_epochs:
        if stop_check:
            stop_check()
        if (root / "stop.flag").exists():
            raise StopRequested("Stop flag: retained the last complete epoch")
        for name, expected in read(root / "source_manifest.json").items():
            if digest(Path(__file__).parent / name) != expected:
                raise RuntimeError("Frozen study code changed; current checkpoints retained")
        state({"phase": stage.upper(), "variant": variant, "seed": seed, "epoch": epoch + 1,
               "epochs": epochs, "run": str(directory)})
        started = time.perf_counter()
        warmup = 3 if stage == "pilot" else 5
        cosine = .5 * (1 + math.cos(math.pi * max(0, epoch - warmup) / max(1, epochs - warmup)))
        lr = config["training"]["learning_rate"] * min(1, (epoch + 1) / warmup) * cosine
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(seed * 10000 + epoch),
                            num_workers=0, pin_memory=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss, structures = 0., 0
        for step, batch in enumerate(loader):
            geometry, controls, target, logs, physics, valid = batch_inputs(batch, scales, device)
            window = min(config["training"]["accumulation"], len(loader) - (step // config["training"]["accumulation"]) * config["training"]["accumulation"])
            with torch.autocast(device.type, dtype=torch.float16, enabled=config["training"]["amp"] and device.type == "cuda"):
                prediction = model(geometry, controls)
                loss = losses(prediction, target, logs, physics, valid, scales, config)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; previous checkpoint retained")
            scaler.scale(loss / window).backward()
            if (step + 1) % config["training"]["accumulation"] == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(geometry); structures += len(geometry)
        metrics, _ = evaluate([model], snapshot, "validation", config)
        epoch_seconds = time.perf_counter() - started; elapsed += epoch_seconds
        improved = metrics["score"] < best_score
        if improved:
            best_score, best_epoch, stale = metrics["score"], epoch, 0
        else:
            stale += 1
        entry = {"epoch": epoch + 1, "train_loss": total_loss / structures, "lr": lr,
                 "epoch_s": epoch_seconds, **metrics}
        history.append(entry)
        checkpoint = {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "grad_scaler": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                      "best_score": best_score, "stale": stale, "elapsed_s": elapsed, "history": history,
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                      "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        if improved:
            atomic_checkpoint(directory / "best.pt", checkpoint)
        atomic_checkpoint(last_path, checkpoint)
        write_csv(directory / "history.csv", history)
        save(directory / "progress.json", {"epoch": epoch + 1, "best_epoch": best_epoch + 1,
                                            "best_score": best_score, "stale": stale, "metrics": metrics,
                                            "elapsed_s": elapsed, "peak_gpu_GiB": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"stage": stage, "variant": variant, "seed": seed, "epoch": epoch + 1,
                              "score": metrics["score"], "elapsed_s": elapsed}), flush=True)
        if stale >= patience:
            break
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    metrics, predictions = evaluate([model], snapshot, "validation", config)
    write_csv(directory / "validation_predictions.csv", predictions)
    result = {"identity": identity, "variant": variant, "seed": seed, "status": "COMPLETE",
              "checkpoint": str(directory / "best.pt"), "checkpoint_sha256": digest(directory / "best.pt"),
              "best_epoch": best["epoch"] + 1, "epochs_ran": len(history), "validation": metrics,
              "parameters": sum(p.numel() for p in model.parameters()), "elapsed_s": elapsed,
              "median_epoch_s": float(np.median([h["epoch_s"] for h in history[1:]] or [history[0]["epoch_s"]])),
              "peak_gpu_GiB": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
              "test_evaluated": False}
    save(directory / "result.json", result)
    del model, optimizer, scaler, checkpoint, best, dataset, loader
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"completed": stage, "variant": variant, "seed": seed, "validation": metrics}), flush=True)
    return result


def mean_baseline(snapshot, config):
    dataset = CachedDataset(snapshot, "train")
    summed = np.zeros((4, 64, 64, 64), dtype=np.float64)
    counts = np.zeros((64, 64, 64), dtype=np.float64)
    for i in dataset.indices:
        valid = dataset.arrays["valid"][i].astype(bool)
        summed += dataset.arrays["field"][i] * valid
        counts += valid
    field = summed / np.maximum(counts, 1)
    field[:3] /= .001
    field[3] /= dataset.meta["scaler"]["pressure_ref_Pa"]
    return evaluate([], snapshot, "validation", config,
                    baseline=(torch.from_numpy(field.astype(np.float32)), None))


def select_variant(results, config):
    names, seeds = config["variants"], config["pilot"]["seeds"]
    table = []
    lookup = {(r["variant"], r["seed"]): r for r in results}
    tasks = ("relative_velocity_l2", "relative_pressure_l2", "lnK_mae", "lnDP_mae")
    for name in names:
        runs = [lookup[(name, seed)] for seed in seeds]
        row = {"variant": name, "mean_score": float(np.mean([r["validation"]["score"] for r in runs])),
               "std_score": float(np.std([r["validation"]["score"] for r in runs], ddof=1)),
               "median_epoch_s": float(np.median([r["median_epoch_s"] for r in runs])),
               "parameters": runs[0]["parameters"]}
        row.update({key: float(np.mean([r["validation"][key] for r in runs])) for key in tasks})
        table.append(row)
    plain = next(row for row in table if row["variant"] == "none")
    eligible = [plain]
    for row in table:
        row["relative_score_gain"] = 1 - row["mean_score"] / plain["mean_score"]
        row["wins_both_seeds"] = all(lookup[(row["variant"], seed)]["validation"]["score"]
                                     < lookup[("none", seed)]["validation"]["score"] for seed in seeds)
        row["maximum_task_regression"] = max(row[key] / max(plain[key], 1e-12) - 1 for key in tasks)
        row["eligible"] = row["variant"] == "none" or (
            row["relative_score_gain"] >= config["selection"]["minimum_relative_gain_for_extra_complexity"]
            and row["wins_both_seeds"] and row["maximum_task_regression"] <= config["selection"]["maximum_task_regression"])
        if row["variant"] != "none" and row["eligible"]:
            eligible.append(row)
    best = min(eligible, key=lambda row: row["mean_score"])
    close = [row for row in eligible if row["mean_score"] <= best["mean_score"] * 1.01]
    chosen = min(close, key=lambda row: row["median_epoch_s"])
    return {"selected_variant": chosen["variant"], "selection_split": "validation", "test_evaluated": False,
            "rule": config["selection"], "table": table, "pilot_seeds": seeds,
            "note": "Internal development results; not an architecture-ranking claim for the manuscript."}


def verify_cache(snapshot):
    cache = Path(snapshot) / "cache"
    for name, expected in read(cache / "manifest.json")["arrays"].items():
        if digest(cache / f"{name}.npy") != expected:
            raise RuntimeError("Frozen cache hash mismatch")


def check_resources(config, state, root):
    while True:
        ram = psutil.virtual_memory().available / 2**30
        free_disk = psutil.disk_usage(str(root)).free / 2**30
        if ram >= config["resources"]["minimum_free_RAM_GiB"] and free_disk >= config["resources"]["minimum_free_disk_GiB"]:
            return
        if (root / "stop.flag").exists():
            raise StopRequested("Stopped while waiting for resources")
        state({"phase": "WAITING_FOR_RESOURCES", "available_RAM_GiB": ram, "free_disk_GiB": free_disk})
        time.sleep(15)


def formal_test(root, snapshot, config, selected, runs, state):
    protocol = {"selected_variant": selected, "config_sha256": digest(root / "config.json"),
                "source_manifest_sha256": digest(root / "source_manifest.json"),
                "snapshot_sha256": digest(snapshot / "snapshot.json"),
                "cache_sha256": digest(snapshot / "cache/manifest.json"),
                "checkpoints": [{"seed": run["seed"], "path": run["checkpoint"],
                                  "sha256": run["checkpoint_sha256"]} for run in runs],
                "ensemble": "arithmetic mean fields; mean log K and log delta-p (geometric mean scalars)",
                "test_used_for_selection": False, "created_epoch_time": time.time()}
    freeze_path = root / "formal_protocol_frozen.json"
    if freeze_path.exists():
        previous = read(freeze_path)
        protocol["created_epoch_time"] = previous["created_epoch_time"]
        if previous != protocol:
            raise RuntimeError("Formal protocol/checkpoints changed; test evaluation blocked")
    else:
        save(freeze_path, protocol)
    protocol_hash = digest(freeze_path)
    summary = []
    for run in runs:
        state({"phase": "FORMAL_TEST", "seed": run["seed"], "protocol_sha256": protocol_hash})
        destination = root / "formal_test" / f"seed{run['seed']}"
        if (destination / "metrics.json").exists():
            saved = read(destination / "metrics.json")
            if saved["protocol_sha256"] != protocol_hash:
                raise RuntimeError("Test report is bound to a different frozen protocol")
            summary.append(saved); continue
        if digest(run["checkpoint"]) != run["checkpoint_sha256"]:
            raise RuntimeError("Selected checkpoint hash mismatch")
        model = initialize_model(config, selected, run["seed"])
        model.load_state_dict(torch.load(run["checkpoint"], map_location="cpu", weights_only=False)["model"])
        metrics, rows = evaluate([model], snapshot, "test", config, destination / "predictions")
        write_csv(destination / "per_structure.csv", rows)
        saved = {"seed": run["seed"], "protocol_sha256": protocol_hash, "metrics": metrics}
        save(destination / "metrics.json", saved); summary.append(saved)
        del model; gc.collect(); torch.cuda.empty_cache()
    destination = root / "formal_test" / "ensemble"
    if not (destination / "metrics.json").exists():
        models = []
        for run in runs:
            model = initialize_model(config, selected, run["seed"])
            model.load_state_dict(torch.load(run["checkpoint"], map_location="cpu", weights_only=False)["model"])
            models.append(model)
        state({"phase": "FORMAL_TEST_ENSEMBLE", "protocol_sha256": protocol_hash})
        metrics, rows = evaluate(models, snapshot, "test", config, destination / "predictions")
        write_csv(destination / "per_structure.csv", rows)
        save(destination / "metrics.json", {"protocol_sha256": protocol_hash, "metrics": metrics})
        del models, model; gc.collect(); torch.cuda.empty_cache()
    ensemble = read(destination / "metrics.json")
    if ensemble["protocol_sha256"] != protocol_hash:
        raise RuntimeError("Ensemble report is bound to a different protocol")
    statistics = {}
    for key in summary[0]["metrics"]:
        values = [row["metrics"][key] for row in summary]
        if all(value is not None for value in values):
            statistics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    save(root / "formal_summary.json", {"status": "COMPLETE", "selected_variant": selected,
                                         "protocol_sha256": protocol_hash, "seeds": summary,
                                         "seed_statistics": statistics, "ensemble": ensemble,
                                         "data_quality_scope": "development CFD labels, production_valid flags unchanged"})


def run_study(root, selection_only=False):
    root = root.resolve()
    config = read(root / "config.json")
    torch.set_num_threads(config["training"]["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("This study is configured for the available CUDA GPU")
    torch.cuda.set_per_process_memory_fraction(config["resources"]["gpu_memory_fraction"])
    record = {"status": "RUNNING", "pid": os.getpid(), "created": psutil.Process().create_time(),
              "config_sha256": digest(root / "config.json"), "started_epoch_time": time.time()}

    def state(values):
        record.update(values, updated_epoch_time=time.time())
        save(root / "state.json", record)

    with study_lock(root):
        try:
            if (root / "state.json").exists():
                previous = read(root / "state.json")
                if previous["config_sha256"] != record["config_sha256"]:
                    raise RuntimeError("Study configuration changed; create a new study")
            state({"phase": "WAITING_FOR_PREPARATION"})
            while True:
                if (root / "stop.flag").exists():
                    raise StopRequested("Stop requested")
                if (root / "preparation.json").exists():
                    preparation = read(root / "preparation.json")
                    if preparation["status"] == "READY":
                        break
                    if preparation["status"] == "FAILED_RETAINED":
                        raise RuntimeError(preparation["error"])
                    try:
                        owner = psutil.Process(preparation["pid"])
                        if abs(owner.create_time() - preparation["created"]) > .1:
                            raise RuntimeError("Snapshot preparer exited; retained its partial artifacts")
                    except psutil.NoSuchProcess as error:
                        raise RuntimeError("Snapshot preparer exited before completion") from error
                time.sleep(3)
            expected_sources = read(root / "source_manifest.json")
            for name, expected in preparation["preprocessing_sources"].items():
                if expected_sources[name] != expected:
                    raise RuntimeError("Preprocessing and frozen training sources differ")
            snapshot = Path(config.get("pilot_snapshot", str(root / "pilot_snapshot")))
            verify_cache(snapshot)
            if not (root / "pilot_mean_baseline.json").exists():
                state({"phase": "PILOT_MEAN_BASELINE"})
                metrics, rows = mean_baseline(snapshot, config)
                save(root / "pilot_mean_baseline.json", metrics)
                write_csv(root / "pilot_mean_baseline_predictions.csv", rows)
            results = []
            for seed in config["pilot"]["seeds"]:
                for variant in config["variants"]:
                    check_resources(config, state, root)
                    results.append(train_run(root, snapshot, config, variant, seed,
                                             config["pilot"]["epochs"], config["pilot"]["patience"], "pilot", state))
            selection = select_variant(results, config)
            save(root / "selection.json", selection)
            write_csv(root / "internal_validation_comparison.csv", selection["table"])
            state({"phase": "SELECTED", "selected_variant": selection["selected_variant"], "test_evaluated": False})
            print(json.dumps({"selection": selection}), flush=True)
            if selection_only or not config["formal"]["auto_when_ready"]:
                state({"status": "SELECTION_COMPLETE", "phase": "SELECTED"}); return
            source = Path(config["source_dataset"])
            target = config["formal"]["minimum_exported_samples"]
            while not (root / "formal_snapshot/snapshot.json").exists():
                if (root / "stop.flag").exists():
                    raise StopRequested("Stopped before formal snapshot")
                manifest = read(source / "manifest.json")
                splits = read(source / "splits.json")
                count = manifest.get("labels", 0)
                coherent = count == sum(len(v) for v in splits.values())
                if count >= target and coherent:
                    break
                state({"phase": "WAITING_FOR_FULL_DATA", "selected_variant": selection["selected_variant"],
                       "exported_samples": count, "required_samples": target})
                time.sleep(30)
            check_resources(config, state, root)
            state({"phase": "FORMAL_SNAPSHOT"})
            full = root / "formal_snapshot"
            full_meta = freeze_snapshot(source, full, train_limit=None, include_test=True)
            if full_meta["source_manifest"]["labels"] != sum(full_meta["counts"].values()):
                raise RuntimeError("Release changed during snapshot metadata capture; retained for review")
            save(root / "method_frozen.json", {"selected_variant": selection["selected_variant"],
                                                "selection_sha256": digest(root / "selection.json"),
                                                "config_sha256": digest(root / "config.json"),
                                                "test_used_for_selection": False})
            state({"phase": "FORMAL_PREPROCESS", "formal_counts": full_meta["counts"]})
            prepare_cache(full, lambda values: state({**values, "phase": "FORMAL_PREPROCESS"}))
            verify_cache(full)
            formal_runs = []
            for seed in config["formal"]["seeds"]:
                check_resources(config, state, root)
                formal_runs.append(train_run(root, full, config, selection["selected_variant"], seed,
                                             config["formal"]["epochs"], config["formal"]["patience"], "formal", state))
            formal_test(root, full, config, selection["selected_variant"], formal_runs, state)
            state({"status": "COMPLETE", "phase": "COMPLETE", "test_evaluated": True})
        except StopRequested as error:
            state({"status": "STOPPED_RETAINED", "error": str(error)})
            print(str(error), flush=True)
        except BaseException:
            state({"status": "FAILED_RETAINED", "error": traceback.format_exc()})
            raise


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    run_study(args.study, args.selection_only)


if __name__ == "__main__":
    main()
