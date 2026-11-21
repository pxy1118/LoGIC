"""Immutable, train-only-normalized snapshots of the growing CFD release."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset

FILES = ("sample.json", "features.json", "controls.npz", "voxel.npz",
         "field.npz", "metrics.json", "quality.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        stream.flush(); os.fsync(stream.fileno())
    for attempt in range(7):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.05 * 2 ** attempt)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_ids(samples, ids, count, seed):
    if count is None or count >= len(ids):
        return sorted(ids)
    groups = defaultdict(list)
    for sid in ids:
        info = read(samples / str(sid) / "sample.json")
        groups[(tuple(info["topology_family"]), info["kind"])].append(sid)
    if count < len(groups):
        raise ValueError("Subset is smaller than its strata")
    keys = sorted(groups)
    allocation = {key: max(1, int(len(groups[key]) * count / len(ids))) for key in keys}
    while sum(allocation.values()) > count:
        key = max((key for key in keys if allocation[key] > 1),
                  key=lambda k: allocation[k] - len(groups[k]) * count / len(ids))
        allocation[key] -= 1
    while sum(allocation.values()) < count:
        key = max((key for key in keys if allocation[key] < len(groups[key])),
                  key=lambda k: len(groups[k]) * count / len(ids) - allocation[k])
        allocation[key] += 1
    selected = []
    for key in keys:
        ranked = sorted(groups[key], key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest())
        selected.extend(ranked[:allocation[key]])
    return sorted(selected)


def release_membership(source):
    """Read a coherent membership without consuming any response values."""
    source = Path(source)
    first = (source / "manifest.json").read_bytes()
    split_bytes = (source / "splits.json").read_bytes()
    if first != (source / "manifest.json").read_bytes():
        raise ValueError("Release metadata are being updated")
    manifest, splits = json.loads(first), json.loads(split_bytes)
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test membership")
    all_ids = [sid for ids in splits.values() for sid in ids]
    if any(type(sid) is not int for sid in all_ids) or len(set(all_ids)) != len(all_ids):
        raise ValueError("Source split IDs must be unique integers")
    if manifest["labels"] != len(all_ids) or manifest["splits"] != {k: len(v) for k, v in splits.items()}:
        raise ValueError("Manifest and split membership differ")
    return manifest, splits, first, split_bytes


def inspect_release(source, minimum_samples=500):
    try:
        manifest, splits, _, _ = release_membership(source)
        missing = []
        for ids in splits.values():
            for sid in ids:
                for name in FILES:
                    path = Path(source) / "samples" / str(sid) / name
                    if not path.is_file() or not path.stat().st_size:
                        missing.append(f"{sid}/{name}")
        return {"ready": manifest["labels"] >= minimum_samples and not missing,
                "exported_samples": manifest["labels"], "required_samples": minimum_samples,
                "splits": {k: len(v) for k, v in splits.items()}, "missing_files": missing,
                "synthetic_fixture": bool(manifest.get("synthetic_fixture", False))}
    except (OSError, ValueError, KeyError) as error:
        return {"ready": False, "exported_samples": None, "required_samples": minimum_samples,
                "reason": str(error)}


def freeze_snapshot(source, destination, train_limit=None, include_test=False, seed=20261113, progress=None):
    source, destination = Path(source), Path(destination)
    final = destination / "snapshot.json"
    if final.exists():
        meta = read(final)
        recipe = {"train_limit": train_limit, "include_test": include_test, "seed": seed}
        if meta.get("recipe", recipe) != recipe or meta["source"] != str(source.resolve()):
            raise RuntimeError("Snapshot inputs or recipe changed; use a new study")
        if meta["status"] == "READY":
            for name, expected in meta["files"].items():
                if digest(destination / name) != expected:
                    raise RuntimeError("Frozen snapshot file changed")
            return meta
        if meta["status"] != "COPYING":
            raise RuntimeError("Unrecognized retained snapshot state")
        chosen = meta["splits"]
    else:
        destination.mkdir(parents=True, exist_ok=False)
        manifest, splits, manifest_bytes, split_bytes = release_membership(source)
        chosen = {"train": stratified_ids(source / "samples", splits["train"], train_limit, seed),
                  "validation": sorted(splits["validation"]),
                  "test": sorted(splits["test"]) if include_test else []}
        meta = {"status": "COPYING", "source": str(source.resolve()), "source_manifest": manifest,
                "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "source_splits_sha256": hashlib.sha256(split_bytes).hexdigest(),
                "source_test_ids": splits["test"], "splits": chosen,
                "recipe": {"train_limit": train_limit, "include_test": include_test, "seed": seed},
                "test_response_copied": include_test, "subset_seed": seed, "files": {}}
        save(final, meta)
    seen_groups = defaultdict(set)
    for split, ids in chosen.items():
        for sid in ids:
            origin = source / "samples" / str(sid)
            output = destination / "samples" / str(sid)
            output.mkdir(parents=True, exist_ok=True)
            for name in FILES:
                path = origin / name
                target = output / name
                relative = target.relative_to(destination).as_posix()
                expected = meta["files"].get(relative)
                if expected and target.exists():
                    if digest(target) != expected or digest(path) != expected:
                        raise RuntimeError("Snapshot/source changed during resume; use a new study")
                    continue
                for attempt in range(4):
                    before = path.stat()
                    data = path.read_bytes()
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                        break
                    time.sleep(.2)
                else:
                    raise RuntimeError(f"Source changed during snapshot: {path}")
                temporary = target.with_name(target.name + ".partial")
                with temporary.open("wb") as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, target)
                meta["files"][relative] = hashlib.sha256(data).hexdigest()
            info = read(output / "sample.json")
            if info["split"] != split or info["sample_id"] != sid:
                raise ValueError("Sample identity differs from release membership")
            seen_groups[info["geometry_group"]].add(split)
            save(final, meta)
            if progress:
                progress({"phase": "SNAPSHOT", "sample_id": sid, "copied_files": len(meta["files"]),
                          "total_files": sum(len(v) for v in chosen.values()) * len(FILES)})
    if any(len(v) > 1 for v in seen_groups.values()):
        raise ValueError("A geometry group crosses snapshot splits")
    meta["status"] = "READY"
    meta["counts"] = {k: len(v) for k, v in chosen.items()}
    save(final, meta)
    return meta


def overlap_matrix(origin, spacing, source_count, target_nodes):
    source_edges = origin - spacing / 2 + np.arange(source_count + 1) * spacing
    nodes = np.asarray(target_nodes, dtype=float)
    target_edges = np.r_[nodes[0], (nodes[:-1] + nodes[1:]) / 2, nodes[-1]]
    widths = np.diff(target_edges)
    if np.any(widths <= 0):
        raise ValueError("Non-increasing query grid")
    overlap = np.maximum(0, np.minimum(target_edges[1:, None], source_edges[None, 1:])
                         - np.maximum(target_edges[:-1, None], source_edges[None, :-1]))
    return csr_matrix((overlap / widths[:, None]).astype(np.float32))


def geometry_features(voxel, origin, spacing, grid_origin, grid_spacing, grid_shape):
    """Initial-only inputs; field values and label masks are not arguments."""
    voxel = np.asarray(voxel, dtype=bool)
    origin, spacing = np.asarray(origin), np.asarray(spacing)
    nodes = [float(grid_origin[k]) + np.arange(grid_shape[k]) * float(grid_spacing[k]) for k in range(3)]
    fraction = voxel.astype(np.float32)
    for axis in range(3):
        matrix = overlap_matrix(origin[axis], spacing[axis], voxel.shape[axis], nodes[axis])
        moved = np.moveaxis(fraction, axis, 0)
        pooled = matrix @ moved.reshape(moved.shape[0], -1)
        fraction = np.moveaxis(np.asarray(pooled).reshape(len(nodes[axis]), *moved.shape[1:]), 0, axis)
    fluid_fraction = np.clip(1 - fraction, 0, 1)
    grid = np.stack(np.meshgrid(*nodes, indexing="ij"))
    indices = (grid - origin[:, None, None, None]) / spacing[:, None, None, None]
    distance = (distance_transform_edt(~voxel, sampling=spacing)
                - distance_transform_edt(voxel, sampling=spacing)).astype(np.float32)
    signed = map_coordinates(distance, indices, order=1, mode="nearest", prefilter=False)
    low_z = origin[2] - spacing[2] / 2
    high_z = low_z + voxel.shape[2] * spacing[2]
    outside_z = np.maximum(low_z - grid[2], np.maximum(grid[2] - high_z, 0))
    outside = outside_z > 0
    signed[outside] = np.sqrt(np.maximum(signed[outside], 0) ** 2 + outside_z[outside] ** 2)
    wall_distance = np.minimum.reduce([grid[0] - nodes[0][0], nodes[0][-1] - grid[0],
                                       grid[1] - nodes[1][0], nodes[1][-1] - grid[1]])
    signed = np.minimum(signed, wall_distance)
    features = np.concatenate((fluid_fraction[None], np.clip(signed / .5, -1, 1)[None],
                               2 * grid / 7.5 - 1, outside[None].astype(np.float32)), axis=0)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite initial-geometry inputs")
    lengths_m = np.asarray(grid_spacing, dtype=float) * (np.asarray(grid_shape) - 1) * .001
    return features.astype(np.float32), float(lengths_m[0] * lengths_m[1])


def initial_inputs(sample, grid_origin, grid_spacing, grid_shape=(64, 64, 64)):
    from architecture import encode_controls
    sample = Path(sample)
    started = time.perf_counter()
    with np.load(sample / "voxel.npz") as source:
        inputs, area = geometry_features(source["voxel"], source["origin_mm"], source["spacing_mm"],
                                        grid_origin, grid_spacing, grid_shape)
    with np.load(sample / "controls.npz") as source:
        controls = encode_controls({key: source[key] for key in source.files})
    return inputs, controls, area, time.perf_counter() - started


def prepare_cache(snapshot, progress=None):
    snapshot = Path(snapshot)
    meta = read(snapshot / "snapshot.json")
    cache = snapshot / "cache"
    if (cache / "manifest.json").exists():
        result = read(cache / "manifest.json")
        if result["status"] != "READY" or result["snapshot_sha256"] != digest(snapshot / "snapshot.json"):
            raise RuntimeError("Cache does not match its snapshot")
        return result
    cache.mkdir(exist_ok=True)
    rows = [(split, sid) for split, ids in meta["splits"].items() for sid in ids]
    count = len(rows)
    arrays = {}
    specifications = {"geometry": ((count, 6, 64, 64, 64), np.float32),
                      "field": ((count, 4, 64, 64, 64), np.float32),
                      "valid": ((count, 64, 64, 64), np.uint8),
                      "controls": ((count, 13, 3, 3, 3), np.float32),
                      "logs": ((count, 2), np.float64), "physics": ((count, 2), np.float64)}
    cursor = cache / "progress.json"
    snapshot_hash = digest(snapshot / "snapshot.json")
    if cursor.exists():
        journal = read(cursor)
        if journal["snapshot_sha256"] != snapshot_hash or journal["rows"] != rows_as_json(rows):
            raise RuntimeError("Partial cache belongs to different snapshot members")
    else:
        if any((cache / f"{name}.npy").exists() for name in specifications):
            raise RuntimeError("Untracked cache arrays retained; use a new study")
        journal = {"snapshot_sha256": snapshot_hash, "rows": rows_as_json(rows),
                   "completed": 0, "pressure_rms": [], "timings": [], "input_timings": []}
        save(cursor, journal)
    for name, (shape, dtype) in specifications.items():
        path = cache / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", shape=shape, dtype=dtype)
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
            raise RuntimeError("Partial cache array shape/dtype changed")
    pressure_rms, timings, input_timings = journal["pressure_rms"], journal["timings"], journal["input_timings"]
    for i in range(journal["completed"], count):
        split, sid = rows[i]
        started = time.perf_counter()
        sample = snapshot / "samples" / str(sid)
        for name in FILES:
            path = sample / name
            if digest(path) != meta["files"][path.relative_to(snapshot).as_posix()]:
                raise ValueError("Frozen sample file hash differs")
        with np.load(sample / "field.npz") as source:
            valid = source["mask"].astype(bool)
            field = np.stack([source[k] for k in ("u", "v", "w", "p")])
            if valid.shape != (64, 64, 64) or not valid.any() or not np.isfinite(field[:, valid]).all():
                raise ValueError(f"Invalid field interface for sample {sid}")
            grid_origin, grid_spacing = source["origin_mm"], source["spacing_mm"]
        inputs, controls, area, input_time = initial_inputs(sample, grid_origin, grid_spacing, valid.shape)
        metrics = read(sample / "metrics.json")
        k, dp, flux = [float(metrics[key]) for key in ("K_domain_app_z_m2", "delta_p_domain_Pa", "Q_m3_s")]
        if not all(np.isfinite(v) and v > 0 for v in (k, dp, flux, area)):
            raise ValueError("Scalar target interface requires positive finite K/delta-p/Q")
        field[3] -= float(metrics["pressure_outlet_Pa"])
        if split == "train":
            pressure_rms.append(float(np.mean(field[3, valid].astype(float) ** 2)))
        field[:, ~valid] = 0
        arrays["geometry"][i], arrays["field"][i], arrays["valid"][i] = inputs, field, valid
        arrays["controls"][i], arrays["logs"][i], arrays["physics"][i] = controls, np.log([k, dp]), [flux, area]
        timings.append(time.perf_counter() - started)
        input_timings.append(input_time)
        for array in arrays.values():
            array.flush()
        journal.update(completed=i + 1, pressure_rms=pressure_rms, timings=timings, input_timings=input_timings)
        save(cursor, journal)
        if progress:
            progress({"phase": "PREPROCESS", "sample_id": sid, "completed": i + 1, "total": count})
    train_rows = [i for i, (split, _) in enumerate(rows) if split == "train"]
    log_train = np.asarray(arrays["logs"][train_rows])
    control_train = np.asarray(arrays["controls"][train_rows])
    scaler = {"log_mean": log_train.mean(0).tolist(), "log_scale": np.maximum(log_train.std(0), 1e-6).tolist(),
              "control_mean": control_train.mean((0, 2, 3, 4)).tolist(),
              "control_scale": np.maximum(control_train.std((0, 2, 3, 4)), 1e-6).tolist(),
              "pressure_ref_Pa": max(float(np.sqrt(np.mean(pressure_rms))), 1e-6),
              "velocity_ref_m_s": .001, "fitted_split": "train", "fitted_ids": meta["splits"]["train"]}
    for value in arrays.values():
        value.flush()
    arrays.clear()
    result = {"status": "READY", "snapshot_sha256": digest(snapshot / "snapshot.json"),
              "rows": [{"split": split, "sample_id": sid} for split, sid in rows], "scaler": scaler,
              "preprocess_time_s": {"total": sum(timings), "median": float(np.median(timings)),
                                     "p95": float(np.quantile(timings, .95))},
              "input_construction_time_s": {"median": float(np.median(input_timings)),
                                             "p95": float(np.quantile(input_timings, .95)),
                                             "scope": "initial voxel/control loading and geometry feature construction only"},
              "arrays": {name: digest(cache / f"{name}.npy") for name in specifications}}
    save(cache / "manifest.json", result)
    return result


def rows_as_json(rows):
    return [{"split": split, "sample_id": sid} for split, sid in rows]


class CachedDataset(Dataset):
    def __init__(self, snapshot, split):
        self.cache = Path(snapshot) / "cache"
        self.meta = read(self.cache / "manifest.json")
        self.indices = [i for i, row in enumerate(self.meta["rows"]) if row["split"] == split]
        self.arrays = {name: np.load(self.cache / f"{name}.npy", mmap_mode="r") for name in self.meta["arrays"]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        result = {name: torch.from_numpy(np.array(array[i], copy=True)) for name, array in self.arrays.items()}
        result["sample_id"] = self.meta["rows"][i]["sample_id"]
        return result
