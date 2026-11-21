"""Frozen, serial, geometry-only validation of the current LoGIC generator."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path); return
        except PermissionError:
            if attempt == 7: raise
            time.sleep(.04 * 2**attempt)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def plan_cases():
    rows = []
    def add(name, section, topology, density=.30, rotation=None, **extra):
        rows.append(dict(case_id=name, section=section, resolution=240, topology=topology,
                         density=density, rotation=rotation, **extra))
    for angle in (0, 15, 30, 45):
        add(f"orientation_constant_{angle}", "orientation", "Primitive", rotation=dict(mode="constant", angles=[0, angle, 0]))
    for angle in (15, 30, 40, 60):
        add(f"orientation_gradient_z_{angle}", "orientation", "Primitive",
            rotation=dict(mode="gradient", field_axis=2, rotation_axis=1, maximum=angle),
            stress_case=angle == 60)
    for axis in (0, 1):
        add(f"orientation_gradient_{'xyz'[axis]}_30", "orientation", "Primitive",
            rotation=dict(mode="gradient", field_axis=axis, rotation_axis=1, maximum=30))
    add("orientation_twist_z_30", "orientation", "Primitive",
        rotation=dict(mode="gradient", field_axis=2, rotation_axis=2, maximum=30))
    add("orientation_multiaxis", "orientation", "Primitive", rotation=dict(mode="multiaxis", maximum=15))
    for topology in ("Gyroid", "Diamond", "Primitive", "Gyroid+Diamond"):
        for value in (.25, .30, .35, .40, .50):
            add(f"uniform_{topology.replace('+', '_')}_{int(value*100)}", "uniform_density", topology, value)
    add("topology_gradient", "coupled", "G_to_D")
    for axis in range(3):
        add(f"density_gradient_{'xyz'[axis]}", "coupled", "Gyroid+Diamond",
            dict(mode="gradient", axis=axis, minimum=.25, maximum=.45))
    add("coupled_three_controls", "coupled", "G_to_D", dict(mode="gradient", axis=2, minimum=.25, maximum=.45),
        rotation=dict(mode="gradient", field_axis=2, rotation_axis=1, maximum=40))
    add("coupled_three_topologies", "coupled", "G_D_P_z", rotation=dict(mode="gradient", field_axis=2, rotation_axis=1, maximum=30))
    assert len(rows) == 38
    for name in ("orientation_gradient_z_40", "uniform_Gyroid_30", "density_gradient_z"):
        row = dict(next(r for r in rows if r["case_id"] == name))
        row.update(case_id=name + "_n120", resolution=120, resolution_reference=name)
        rows.append(row)
    return rows


def controls(case):
    order = ["Gyroid", "Diamond", "Primitive"]
    weights = np.zeros((3, 3, 3, 3)); angles = np.zeros((3, 3, 3, 3))
    xyz = np.indices((3, 3, 3)).astype(float) / 2
    topology = case["topology"]
    if topology == "G_to_D":
        weights[..., 0], weights[..., 1] = 1 - xyz[2], xyz[2]
    elif topology == "G_D_P_z":
        weights[:] = np.eye(3)[np.indices((3, 3, 3))[2]]
    else:
        names = topology.split("+")
        for name in names:
            weights[..., order.index(name)] = 1 / len(names)
    rho = case["density"]
    if isinstance(rho, dict):
        density = rho["minimum"] + (rho["maximum"] - rho["minimum"]) * xyz[rho["axis"]]
    else:
        density = np.full((3, 3, 3), rho)
    rotation = case["rotation"]
    if rotation:
        if rotation["mode"] == "constant": angles[:] = rotation["angles"]
        elif rotation["mode"] == "gradient":
            angles[..., rotation["rotation_axis"]] = rotation["maximum"] * xyz[rotation["field_axis"]]
        else:
            for axis in range(3): angles[..., axis] = rotation["maximum"] * (xyz[axis] - .5)
    return weights, density, angles


def stratified_selection(records, count=50):
    groups = defaultdict(list)
    for row in records:
        groups[(tuple(row["topology_family"]), row["kind"])].append(row)
    keys = sorted(groups)
    allocated = {key: max(1, int(len(groups[key]) * count / len(records))) for key in keys}
    while sum(allocated.values()) > count:
        key = max((k for k in keys if allocated[k] > 1), key=lambda k: allocated[k] - len(groups[k]) * count / len(records))
        allocated[key] -= 1
    while sum(allocated.values()) < count:
        key = max((k for k in keys if allocated[k] < len(groups[k])), key=lambda k: len(groups[k]) * count / len(records) - allocated[k])
        allocated[key] += 1
    selected = []
    for key in keys:
        ranked = sorted(groups[key], key=lambda r: hashlib.sha256(f"20261114:{r['original_sample_id']}".encode()).hexdigest())
        selected.extend(ranked[:allocated[key]])
    return sorted(selected, key=lambda r: r["sample_id"])


def initialize(root):
    workspace = Path(__file__).resolve().parent
    project = workspace.parent.parent
    if root.exists(): raise FileExistsError("Study already exists; run it without reinitializing")
    source = root / "source"; source.mkdir(parents=True)
    for name in ("experiment.py", "measurements.py"):
        shutil.copy2(workspace / name, source / name)
    for name in ("tpms_hybrid.py", "orientation_mapping.py"):
        shutil.copy2(project / name, source / name)
    frozen = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    release = Path(os.environ.get("LOGIC_FEA_DATA", project / "data/fea")).resolve()
    snapshot_path = release / "manifest.json"
    records = []
    for entry in read(snapshot_path)["samples"]:
        meta = read(release / entry["path"] / "sample.json")
        records.append(dict(meta, source_hashes=entry["hashes"]))
    records = stratified_selection(records)
    reuse = []
    for row in records:
        files = {name: row["source_hashes"][name] for name in ("sample.json", "controls.npz", "voxel.npz", "features.json")}
        for name, expected in files.items():
            if digest(release / "samples" / str(row["sample_id"]) / name) != expected:
                raise ValueError("Published initial-input hash differs")
        reuse.append({"sample_id": row["sample_id"], "topology_family": row["topology_family"],
                      "kind": row["kind"], "split": row["split"], "geometry_group": row["geometry_group"],
                      "source": str(release / "samples" / str(row["sample_id"])), "files": files})
    geometry = read(release / "config.json")["geometry"]
    protocol = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "purpose": "Section 3.1 geometry control validation; no solver runs or surrogate changes",
                "geometry": geometry, "cases": plan_cases(), "reuse": reuse,
                "source_manifest": frozen, "release_manifest_sha256": digest(snapshot_path),
                "measurement": {"primary_regions_per_axis": 3, "sensitivity_regions_per_axis": 6,
                                "density_error_unit": "percentage points (100 times SVF difference)",
                                "orientation": "full-node original Jacobian diagnostic plus fixed sparse xyz-axis detail",
                                "phase_connectivity": "6-neighbor primary, 26-neighbor sensitivity for reused voxels",
                                "pore_transport": "sum of all pore components spanning both opposite faces; single pore component not required",
                                "replication": "deterministic design grid; regions/nodes are not independent replicates"},
                "scope": {"new_surfaces": "raw closed initial surface, no component removal or smoothing",
                          "reused_geometry": "published initial FE voxel geometry, conditional on completed cohort",
                          "existing_original_surfaces": "not available in retained source-run paths; no old surface-mesh statistics inferred",
                          "responses": "No curve.npz, metrics.json or numerical quality labels are read"},
                "resource": {"workers": 1, "threads": 2, "minimum_available_RAM_GiB": 16,
                             "minimum_free_disk_GiB": 30, "priority": "BELOW_NORMAL", "gpu": "not used by measurements"}}
    save(root / "protocol.json", protocol)
    save(root / "identity.json", {"protocol_sha256": digest(root / "protocol.json")})
    save(root / "environment.json", {"python": sys.version, "packages": {name: importlib.metadata.version(name)
         for name in ("numpy", "scipy", "pyvista", "psutil", "matplotlib", "PyMuPDF")}})
    save(root / "state.json", {"status": "INITIALIZED", "new_cases": len(protocol["cases"]), "reused_cases": len(reuse)})


def verify(root):
    protocol = read(root / "protocol.json")
    if digest(root / "protocol.json") != read(root / "identity.json")["protocol_sha256"]:
        raise RuntimeError("Protocol changed; create another study")
    for name, expected in protocol["source_manifest"].items():
        if digest(root / "source" / name) != expected: raise RuntimeError("Frozen source changed")
    return protocol


def setup_priority():
    process = psutil.Process()
    if os.name == "nt": process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    allowed = process.cpu_affinity()
    process.cpu_affinity(allowed[-2:])
    try:
        import vtk
        vtk.vtkSMPTools.Initialize(2)
    except ImportError:
        pass


def validate_result(path, identity):
    data = read(path)
    if data["protocol_sha256"] != identity: raise RuntimeError("Case belongs to another protocol")
    for name, expected in data["hashes"].items():
        if digest(path.parent / name) != expected: raise RuntimeError("Saved measurement changed")
    return data


def worker(root, case_id):
    setup_priority(); protocol = verify(root)
    identity = digest(root / "protocol.json")
    folder = root / "cases" / case_id; folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists(): validate_result(result_path, identity); return
    started = time.perf_counter()
    try:
        from measurements import density_regions, axial_profile, voxel_integrity, surface_integrity, sample_phase_diagnostics
        case = next((row for row in protocol["cases"] if row["case_id"] == case_id), None)
        if case is None:
            row = next(row for row in protocol["reuse"] if f"reuse_{row['sample_id']}" == case_id)
            source = Path(row["source"])
            for name, expected in row["files"].items():
                if digest(source / name) != expected: raise ValueError("Initial input changed")
            with np.load(source / "voxel.npz") as value:
                mask = value["voxel"].astype(bool); origin = value["origin_mm"]; spacing = value["spacing_mm"]
            with np.load(source / "controls.npz") as value:
                density = value["density"].copy()
            coords = [origin[k] + spacing[k] * np.arange(mask.shape[k]) for k in range(3)]
            regions = density_regions(mask, density, coords)
            save(folder / "density.json", regions)
            result = {"kind": "reused", "case_id": case_id, "sample_id": row["sample_id"],
                      "topology": "+".join(row["topology_family"]), "sample_kind": row["kind"],
                      "integrity": voxel_integrity(mask, sensitivity=True),
                      "density": {k: v for k, v in regions.items() if k != "regions"},
                      "surface": None, "source_hashes": row["files"],
                      "surface_scope": "Original surface unavailable; voxel integrity only; no reconstructed-surface AR substituted"}
        else:
            import pyvista as pv
            import tpms_hybrid as gen
            weights, density, angles = controls(case)
            n = case["resolution"]; geometry = protocol["geometry"]
            span = 3. * geometry["periods"]; bounds = [(-span / 2, span / 2)] * 3
            size = geometry["size_mm"]
            W = gen.generate_weight_matrix(n, *bounds, weights, smooth_sigma=geometry["weight_smoothing_sigma_mm"] / (size / (n - 1)))
            D = gen.generate_density_field_from_3x3x3(n, *bounds, density)
            Q = gen.generate_rotation_field_from_3x3x3(n, *bounds, angles) if case["rotation"] else None
            phase = np.asarray(geometry["phase_offset_periods"]) * 3
            functions = [(lambda x, y, z, f=getattr(gen, name): f(x + phase[0], y + phase[1], z + phase[2]))
                         for name in geometry["topologies"]]
            quality = {}; extra = {}
            original = gen.mapping_diagnostics
            def observe(phases, quaternions, ranges):
                values = original(phases, quaternions, ranges)
                constant = case["rotation"]["angles"] if case["rotation"]["mode"] == "constant" else None
                detail, summary = sample_phase_diagnostics(phases, quaternions, ranges, constant)
                detail["positions_mm"] = (detail.pop("positions_generator") * (size / span) + size / 2)
                np.savez_compressed(folder / "orientation_nodes.npz", **detail)
                extra.update(summary)
                save(folder / "orientation.json", {"full_node": values, "sparse": summary})
                return values
            gen.mapping_diagnostics = observe
            # No axis-error/stretch filtering: stress-case failures remain observations.
            mesh, _, _, mask = gen.create_hybrid_tpms_solid(functions, *bounds, weight_volume=W,
                density_field=D, rotation_field=Q, resolution=n, return_mask=True, rotation_mode="integrated",
                rotation_diagnostics=quality, max_axis_error_deg=None, max_phase_stretch_ratio=None,
                morph_close=False, remove_small=False, smooth=False, verbose=False, surface_boundary_mode="implicit_box")
            mesh = pv.PolyData(mesh.points, mesh.faces).triangulate().clean()
            mesh.points = np.asarray(mesh.points, dtype=np.float64)
            mesh.scale(size / span, inplace=True); mesh.translate((size / 2,) * 3, inplace=True)
            mesh.save(folder / "surface.stl")
            checked = pv.read(folder / "surface.stl")
            surface = surface_integrity(checked, size)
            coords = [np.linspace(0, size, n)] * 3
            region = density_regions(mask, density, coords, size)
            save(folder / "density.json", region)
            region_sensitivity = density_regions(mask, density, coords, size, bins=6)
            save(folder / "density_window_sensitivity.json", region_sensitivity)
            np.savez_compressed(folder / "geometry.npz", mask=mask, origin_mm=np.zeros(3),
                                spacing_mm=np.full(3, size / (n - 1)), weights=weights, density=density, rotation_deg=angles)
            profile = {}
            for axis in range(3):
                values = axial_profile(mask, density, coords, axis, size)
                profile.update({f"{axis}_{k}": v for k, v in values.items()})
            np.savez_compressed(folder / "profiles.npz", **profile)
            mesh_error = 100 * (surface["mesh_svf"] - region["target_volume_mean"]) if surface["mesh_svf"] is not None else None
            result = {"kind": "generated", **case, "integrity": voxel_integrity(mask), "surface": surface,
                      "density": {k: v for k, v in region.items() if k != "regions"} | {"mesh_error_pp": mesh_error,
                                  "regional_mae_pp_half_cell": region_sensitivity["regional_mae_pp"]},
                      "orientation": quality, "orientation_detail": extra,
                      "within_dataset_orientation_limits": bool(not quality or
                          quality["local_z_error_max_deg"] <= geometry["max_axis_error_deg"] and
                          quality["max_phase_stretch"] <= geometry["max_phase_stretch_ratio"] and
                          quality["min_phase_stretch"] >= 1 / geometry["max_phase_stretch_ratio"])}
        result.update(status="COMPLETE", protocol_sha256=identity, elapsed_s=time.perf_counter() - started,
                      peak_RAM_GiB=getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss) / 2**30,
                      hashes={p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.suffix in (".npz", ".stl", ".json") and p.name not in ("result.json", "failure.json")})
        save(result_path, result)
    except BaseException:
        save(folder / "failure.json", {"status": "FAILED_RETAINED", "protocol_sha256": identity,
                                       "elapsed_s": time.perf_counter() - started, "error": traceback.format_exc()})
        raise


@contextmanager
def study_lock(root):
    stream = (root / ".writer.lock").open("a+b")
    if not stream.tell(): stream.write(b"0"); stream.flush()
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            stream.close(); raise RuntimeError("Study already has a writer")
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try: yield
    finally:
        stream.seek(0)
        if os.name == "nt": msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def run_all(root, limit=None):
    with study_lock(root):
        return _run_all(root, limit)


def _run_all(root, limit=None):
    setup_priority(); protocol = verify(root); process = psutil.Process()
    identity = digest(root / "protocol.json")
    record = {"status": "RUNNING", "pid": process.pid, "created": process.create_time(),
              "protocol_sha256": identity, "total": len(protocol["cases"]) + len(protocol["reuse"])}
    planned = [r["case_id"] for r in protocol["cases"]] + [f"reuse_{r['sample_id']}" for r in protocol["reuse"]]
    invocation = 0; completed = 0; failed = 0
    def state(**values):
        record.update(values, updated_epoch_time=time.time()); save(root / "state.json", record)
    try:
        prior = read(root / "state.json")
        if "worker_pid" in prior:
            try:
                owned = psutil.Process(prior["worker_pid"])
                if abs(owned.create_time() - prior["worker_created"]) < .1:
                    command = owned.cmdline()
                    if str(root / "source/experiment.py").lower() not in [arg.lower() for arg in command] or "worker" not in command:
                        raise RuntimeError("Worker receipt refers to an unexpected command")
                    while owned.is_running():
                        state(phase="RECOVERING_ACTIVE_WORKER", worker_pid=owned.pid, worker_created=owned.create_time())
                        try: owned.wait(timeout=10)
                        except psutil.TimeoutExpired: continue
                        break
            except psutil.NoSuchProcess:
                pass
        for case_id in planned:
            if (root / "stop.flag").exists(): state(status="STOPPED_RETAINED"); return
            path = root / "cases" / case_id
            if (path / "result.json").exists(): validate_result(path / "result.json", identity); completed += 1; continue
            if (path / "failure.json").exists():
                if read(path / "failure.json")["protocol_sha256"] != identity: raise RuntimeError("Failure belongs to another protocol")
                failed += 1; continue
            if limit is not None and invocation >= limit:
                state(status="PARTIAL_RETAINED", completed=completed, failed=failed); return
            while psutil.virtual_memory().available / 2**30 < 16 or psutil.disk_usage(str(root)).free / 2**30 < 30:
                state(phase="WAITING_FOR_RESOURCES", completed=completed, failed=failed)
                if (root / "stop.flag").exists(): state(status="STOPPED_RETAINED"); return
                time.sleep(10)
            path.mkdir(parents=True, exist_ok=True)
            state(phase="MEASURING", case_id=case_id, completed=completed, failed=failed)
            with (path / "worker.stdout.log").open("ab") as stdout, (path / "worker.stderr.log").open("ab") as stderr:
                child = subprocess.Popen([sys.executable, str(root / "source/experiment.py"), "worker", "--study", str(root), "--case", case_id],
                                         stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                state(worker_pid=child.pid, worker_created=psutil.Process(child.pid).create_time())
                child.wait()
            invocation += 1
            if (path / "result.json").exists(): validate_result(path / "result.json", identity); completed += 1
            else:
                failed += 1
                if not (path / "failure.json").exists():
                    save(path / "failure.json", {"status": "FAILED_RETAINED", "protocol_sha256": identity,
                         "error": f"Worker exited {child.returncode} without a receipt"})
            print(json.dumps({"case": case_id, "completed": completed, "failed": failed, "total": len(planned)}), flush=True)
        state(status="MEASUREMENTS_COMPLETE", phase="MEASUREMENTS_COMPLETE", completed=completed, failed=failed)
    except BaseException:
        state(status="FAILED_RETAINED", error=traceback.format_exc()); raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--study", type=Path, required=True); parser.add_argument("--case"); parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "worker": worker(args.study.resolve(), args.case)
    else: run_all(args.study.resolve(), args.limit)
