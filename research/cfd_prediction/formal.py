"""Formal, selected-architecture-only CFD experiment and sealed test evaluation."""
from __future__ import annotations

import argparse
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

from data import digest, freeze_snapshot, inspect_release, prepare_cache, read, save, CachedDataset
from experiment import (StopRequested, evaluate, initialize_model, resolve_device, scalar_metrics,
                        study_lock, train_run, verify_cache, write_csv)


class BudgetReached(Exception):
    pass


class TestGateError(RuntimeError):
    pass


def verify_identity(root):
    initialized = read(root / "run.json")
    if digest(root / "config.json") != initialized["config_sha256"]:
        raise RuntimeError("Frozen config changed; create a new run")
    if digest(root / "source_manifest.json") != initialized["source_manifest_sha256"]:
        raise RuntimeError("Frozen source manifest changed; create a new run")
    for name, expected in read(root / "source_manifest.json").items():
        if digest(root / "source" / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}; create a new run")
    if digest(root / "selection_reference.json") != initialized["selection_sha256"]:
        raise RuntimeError("Selected method reference changed")
    return initialized


def completed_training(root, config):
    """Gate evaluation before constructing a test loader or opening test targets."""
    verify_identity(root)
    if config["selected_variant"] != "none":
        raise TestGateError("This formal study accepts the selected plain conditional U-Net only")
    snapshot = root / "snapshot"
    if not (snapshot / "cache/manifest.json").exists():
        raise TestGateError("Frozen dataset/cache are not ready")
    cache_hash = digest(snapshot / "cache/manifest.json")
    verify_cache(snapshot)
    for name, expected in read(snapshot / "snapshot.json")["files"].items():
        if digest(snapshot / name) != expected:
            raise TestGateError("Frozen input/target file changed")
    runs = []
    for seed in config["formal"]["seeds"]:
        path = root / "runs" / f"formal_none_seed{seed}" / "result.json"
        if not path.exists():
            raise TestGateError(f"Seed {seed} has not completed training")
        result = read(path)
        identity = result["identity"]
        expected = {"config_sha256": digest(root / "config.json"), "cache_sha256": cache_hash,
                    "sources_sha256": digest(root / "source_manifest.json"), "variant": "none", "seed": seed,
                    "epochs": config["formal"]["epochs"], "patience": config["formal"]["patience"], "stage": "formal"}
        if result["status"] != "COMPLETE" or identity != expected or result.get("test_evaluated"):
            raise TestGateError("Training receipt is inconsistent with the fixed protocol")
        if result["parameters"] != config["model"]["expected_parameters"]:
            raise TestGateError("Selected architecture parameter count differs")
        if digest(Path(result["checkpoint"])) != result["checkpoint_sha256"]:
            raise TestGateError("Selected checkpoint hash differs")
        runs.append(result)
    return runs


def validate_completion(root):
    receipt = read(root / "completion_manifest.json")
    if receipt["protocol_sha256"] != digest(root / "test_protocol_frozen.json"):
        raise TestGateError("Completion receipt refers to another protocol")
    for name, expected in receipt["artifacts"].items():
        if digest(root / name) != expected:
            raise RuntimeError("Completed report artifact changed")


def representative_ids(snapshot, count):
    """Use only initial sample metadata, before computing any test errors."""
    members = read(snapshot / "snapshot.json")["splits"]["test"]
    groups = {}
    for sid in members:
        info = read(snapshot / "samples" / str(sid) / "sample.json")
        key = "+".join(info["topology_family"])
        groups.setdefault(key, []).append(sid)
    selected = []
    for family in ("Gyroid", "Diamond", "Primitive"):
        if family in groups:
            selected.append(min(groups[family]))
    selected = selected[:count]
    for sid in sorted(members):
        if len(selected) >= count:
            break
        if sid not in selected:
            selected.append(sid)
    return selected


def freeze_test_protocol(root, config):
    runs = completed_training(root, config)
    snapshot = root / "snapshot"
    protocol = {"schema_version": 1, "selected_variant": "none", "test_used_for_selection": False,
                "config_sha256": digest(root / "config.json"), "source_manifest_sha256": digest(root / "source_manifest.json"),
                "snapshot_sha256": digest(snapshot / "snapshot.json"), "cache_sha256": digest(snapshot / "cache/manifest.json"),
                "seeds": config["formal"]["seeds"],
                "checkpoints": [{"seed": result["seed"], "path": result["checkpoint"], "sha256": result["checkpoint_sha256"]} for result in runs],
                "representative_test_ids": representative_ids(snapshot, config["report"]["representative_fields"]),
                "ensemble": "mean velocity/pressure fields and mean log-scalar outputs",
                "created_epoch_time": time.time()}
    path = root / "test_protocol_frozen.json"
    if path.exists():
        saved = read(path)
        protocol["created_epoch_time"] = saved["created_epoch_time"]
        if saved != protocol:
            raise TestGateError("Test protocol or checkpoint selection changed")
    else:
        save(path, protocol)
    return protocol, runs


def validate_evaluation(path, protocol_hash):
    result = read(path / "metrics.json")
    if result["protocol_sha256"] != protocol_hash:
        raise TestGateError("Evaluation belongs to another frozen protocol")
    for name, expected in result["artifact_hashes"].items():
        if digest(path / name) != expected:
            raise TestGateError("Saved prediction/report hash changed")
    return result


def evaluate_frozen(root, config, state=lambda values: None, guard=lambda: None):
    protocol, runs = freeze_test_protocol(root, config)
    protocol_hash = digest(root / "test_protocol_frozen.json")
    snapshot = root / "snapshot"
    for tag, members in [(f"seed{run['seed']}", [run]) for run in runs] + [("ensemble", runs)]:
        guard()
        destination = root / "test" / tag
        if (destination / "metrics.json").exists():
            validate_evaluation(destination, protocol_hash)
            continue
        state({"phase": "TEST", "evaluation": tag})
        attempt_path = root / "test_attempts.json"
        attempts = read(attempt_path) if attempt_path.exists() else []
        attempts.append({"tag": tag, "protocol_sha256": protocol_hash, "started_epoch_time": time.time()})
        save(attempt_path, attempts)
        models = []
        for member in members:
            model = initialize_model(config, "none", member["seed"])
            model.load_state_dict(torch.load(member["checkpoint"], map_location="cpu", weights_only=False)["model"])
            models.append(model)
        metrics, rows = evaluate(models, snapshot, "test", config, destination / "predictions",
                                 lambda values: (guard(), state({"phase": "TEST", "evaluation": tag, **values})))
        write_csv(destination / "per_structure.csv", rows)
        artifacts = {path.relative_to(destination).as_posix(): digest(path) for path in destination.rglob("*.npz")}
        artifacts["per_structure.csv"] = digest(destination / "per_structure.csv")
        save(destination / "metrics.json", {"protocol_sha256": protocol_hash, "metrics": metrics, "artifact_hashes": artifacts})
        del models, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return protocol


def benchmark(root, config, guard=lambda: None):
    if not config["benchmark"]["enabled"] or (root / "benchmark.json").exists():
        return
    snapshot = root / "snapshot"
    cache = read(snapshot / "cache/manifest.json")
    dataset = CachedDataset(snapshot, "validation")
    item = dataset[0]
    controls = (item["controls"][None] - torch.tensor(cache["scaler"]["control_mean"])[None, :, None, None, None]) / torch.tensor(cache["scaler"]["control_scale"])[None, :, None, None, None]
    results = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        local = json.loads(json.dumps(config)); local["training"]["device"] = device
        models = []
        for seed in config["formal"]["seeds"]:
            receipt = read(root / "runs" / f"formal_none_seed{seed}" / "result.json")
            model = initialize_model(local, "none", seed).eval()
            model.load_state_dict(torch.load(receipt["checkpoint"], map_location="cpu", weights_only=False)["model"])
            models.append(model)
        geometry, control = item["geometry"][None].to(device), controls.to(device)
        results[device] = {}
        for tag, selected in (("single_model", models[:1]), ("ensemble", models)):
            samples = []
            with torch.inference_mode():
                for i in range(config["benchmark"]["warmup"] + config["benchmark"]["repetitions"]):
                    guard()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda" and config["training"]["amp"]):
                        predictions = [model(geometry, control) for model in selected]
                        torch.stack([out["field_normalized"].float() for out in predictions]).mean(0)
                        torch.stack([out["scalar_standardized"].float() for out in predictions]).mean(0)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    if i >= config["benchmark"]["warmup"]:
                        samples.append(time.perf_counter() - started)
            results[device][tag] = {"median_s": float(np.median(samples)), "p95_s": float(np.quantile(samples, .95)),
                                    "batch_size": 1, "warmup": config["benchmark"]["warmup"], "repetitions": len(samples)}
        del models, model, predictions
        if device == "cuda":
            torch.cuda.empty_cache()
    save(root / "benchmark.json", {"prediction": results, "input_construction": cache["input_construction_time_s"],
                                    "scope": "Forward pass and ensemble aggregation; excludes input construction, transfer, file IO and label access"})


def run_pipeline(root, wait=False, budget_minutes=None, evaluate_only=False):
    root = Path(root).resolve()
    config = read(root / "config.json")
    verify_identity(root)
    torch.set_num_threads(config["training"]["cpu_threads"])
    started = time.monotonic()
    record = {"status": "RUNNING", "pid": os.getpid(), "created": psutil.Process().create_time(),
              "config_sha256": digest(root / "config.json"), "selected_variant": "none", "updated_epoch_time": time.time()}

    def state(values):
        record.update(values, updated_epoch_time=time.time())
        save(root / "state.json", record)

    def guard():
        verify_identity(root)
        if (root / "stop.flag").exists():
            raise StopRequested("Stop requested; completed work/checkpoints retained")
        if budget_minutes is not None and time.monotonic() - started >= budget_minutes * 60:
            raise BudgetReached("Invocation time budget reached; resume the same frozen run")

    def resources():
        while (psutil.virtual_memory().available / 2**30 < config["resources"]["minimum_free_RAM_GiB"]
               or psutil.disk_usage(str(root)).free / 2**30 < config["resources"]["minimum_free_disk_GiB"]):
            guard(); state({"phase": "WAITING_FOR_RESOURCES"}); time.sleep(10)

    with study_lock(root):
        try:
            if (root / "completion_manifest.json").exists() and not evaluate_only:
                validate_completion(root)
                state({"status": "COMPLETE", "phase": "COMPLETE"}); return 0
            if not evaluate_only:
                snapshot = root / "snapshot"
                if not (snapshot / "snapshot.json").exists():
                    while True:
                        guard()
                        readiness = inspect_release(config["source_dataset"], config["minimum_samples"])
                        if readiness.get("synthetic_fixture") and not config.get("allow_synthetic_fixture", False):
                            raise ValueError("Synthetic fixtures are excluded from formal scientific runs")
                        if readiness["ready"]:
                            break
                        state({"phase": "WAITING_FOR_DATA", **readiness})
                        if not wait:
                            state({"status": "DATA_NOT_READY"}); return 3
                        time.sleep(config["wait_poll_seconds"])
                guard(); resources(); state({"phase": "SNAPSHOT"})
                membership = freeze_snapshot(config["source_dataset"], snapshot, include_test=True,
                                              progress=lambda values: (state(values), guard()))
                save(root / "data_frozen.json", {"snapshot_sha256": digest(snapshot / "snapshot.json"),
                                                 "counts": membership["counts"]})
                prepare_cache(snapshot, lambda values: (state(values), guard()))
                verify_cache(snapshot)
                device = resolve_device(config)
                if device.type == "cuda":
                    torch.cuda.set_per_process_memory_fraction(config["resources"]["gpu_memory_fraction"])
                for seed in config["formal"]["seeds"]:
                    guard(); resources()
                    train_run(root, snapshot, config, "none", seed, config["formal"]["epochs"],
                              config["formal"]["patience"], "formal", state, guard)
                guard()
            state({"phase": "TEST_GATE"})
            evaluate_frozen(root, config, state, guard)
            guard(); state({"phase": "BENCHMARK"}); benchmark(root, config, guard)
            guard(); state({"phase": "REPORT"})
            from reporting import make_report
            make_report(root)
            state({"status": "COMPLETE", "phase": "COMPLETE", "test_evaluated": True})
            return 0
        except StopRequested as error:
            state({"status": "STOPPED_RETAINED", "reason": str(error)}); return 0
        except BudgetReached as error:
            state({"status": "BUDGET_REACHED", "reason": str(error)}); return 0
        except BaseException:
            state({"status": "FAILED_RETAINED", "error": traceback.format_exc()}); raise


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--budget-minutes", type=float)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.budget_minutes is not None and (not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0):
        parser.error("Time budget must be finite and positive")
    raise SystemExit(run_pipeline(args.study, args.wait, args.budget_minutes, args.evaluate_only))


if __name__ == "__main__":
    main()
