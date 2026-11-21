"""init/run/status/evaluate/report/predict for the fixed CFD architecture."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

from data import digest, read, save

ROOT = Path(__file__).resolve().parent
DEFAULT = "paper_cfd"
SOURCE_FILES = ["architecture.py", "data.py", "experiment.py", "formal.py", "reporting.py", "predict.py"]


def study_root(name):
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Study name must contain only letters/digits/underscores/hyphens")
    return ROOT / "studies" / name


def initialize(root, config_path):
    config = read(config_path)
    for name in ("source_dataset", "selection_reference"):
        config[name] = str((config_path.resolve().parent / config[name]).resolve())
    if os.environ.get("LOGIC_CFD_DATA"):
        config["source_dataset"] = str(Path(os.environ["LOGIC_CFD_DATA"]).resolve())
    if config["selected_variant"] != "none" or config["model"]["widths"] != [16, 32, 64, 96]:
        raise ValueError("Formal config must retain the selected architecture")
    selected = read(Path(config["selection_reference"]))
    if selected["selected_variant"] != "none" or selected["test_evaluated"]:
        raise ValueError("Selected method reference is inconsistent")
    if (root / "run.json").exists():
        if read(root / "config.json") != config:
            raise ValueError("Run exists with a different config; use a new name")
        from formal import verify_identity
        verify_identity(root)
        return
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"; source.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, source / name)
    save(root / "config.json", config)
    save(root / "selection_reference.json", selected)
    sources = {path.relative_to(source).as_posix(): digest(path) for path in source.rglob("*") if path.is_file()}
    save(root / "source_manifest.json", sources)
    save(root / "run.json", {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              "config_sha256": digest(root / "config.json"),
                              "source_manifest_sha256": digest(root / "source_manifest.json"),
                              "selection_sha256": digest(root / "selection_reference.json"),
                              "purpose": "Selected-model formal CFD experiment"})
    save(root / "state.json", {"status": "INITIALIZED", "phase": "INITIALIZED", "selected_variant": "none"})


def live_owner(root):
    if not (root / "state.json").exists():
        return None
    state = read(root / "state.json")
    if "pid" not in state or "created" not in state:
        if not (root / "launch.json").exists():
            return None
        state = read(root / "launch.json")
    try:
        process = psutil.Process(state["pid"])
        if abs(process.create_time() - state["created"]) > .1:
            return None
        expected = str(root / "source/formal.py").lower()
        if expected not in [arg.lower() for arg in process.cmdline()]:
            raise RuntimeError("Live receipt belongs to another command; no process was touched")
        return process
    except psutil.NoSuchProcess:
        return None


def dispatch(root, args):
    if args.command == "init":
        initialize(root, args.config); print(str(root)); return 0
    if args.command == "status":
        state = read(root / "state.json")
        state["controller_live"] = live_owner(root) is not None
        state["run_directory"] = str(root)
        print(json.dumps(state, ensure_ascii=False, indent=2)); return 0
    if args.command == "stop":
        (root / "stop.flag").write_text("User-requested stop; retain all completed work", encoding="utf-8")
        print("Stop requested. Training completes its current epoch; preprocessing retains its committed cursor."); return 0
    if args.command == "report":
        command = [sys.executable, "-c", "from reporting import make_report; from pathlib import Path; import sys; make_report(Path(sys.argv[1]))", str(root)]
        return subprocess.run(command, cwd=root / "source").returncode
    if args.command == "predict":
        command = [sys.executable, str(root / "source/predict.py"), "--study", str(root), "--inputs", str(args.inputs), "--output", str(args.output)]
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        if args.grid_json:
            command.extend(["--grid-json", str(args.grid_json)])
        return subprocess.run(command).returncode
    if live_owner(root):
        raise RuntimeError("This formal study already has a live controller; use status")
    if (root / "stop.flag").exists():
        if not args.resume:
            raise RuntimeError("Stop marker exists. Use run --resume for an explicit continuation")
        (root / "stop.flag").unlink()
    command = [sys.executable, str(root / "source/formal.py"), "--study", str(root)]
    if args.command == "evaluate":
        command.append("--evaluate-only")
    if args.wait:
        command.append("--wait")
    if args.budget_minutes is not None:
        command.extend(["--budget-minutes", str(args.budget_minutes)])
    if not args.background:
        return subprocess.run(command, cwd=ROOT.parent.parent).returncode
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with (root / f"controller_{suffix}.stdout.log").open("wb") as stdout, (root / f"controller_{suffix}.stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, cwd=ROOT.parent.parent, env=environment, stdout=stdout, stderr=stderr,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    save(root / "launch.json", {"pid": child.pid, "created": psutil.Process(child.pid).create_time(), "command": command,
                               "stdout": str(root / f"controller_{suffix}.stdout.log"), "stderr": str(root / f"controller_{suffix}.stderr.log")})
    print(json.dumps({"pid": child.pid, "study": str(root), "background": True})); return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "run", "status", "evaluate", "report", "predict", "stop"):
        sub = subs.add_parser(name); sub.add_argument("--name", default=DEFAULT)
        if name == "init":
            sub.add_argument("--config", type=Path, default=ROOT / "configs/formal_cfd.json")
        if name in ("run", "evaluate"):
            sub.add_argument("--wait", action="store_true"); sub.add_argument("--background", action="store_true")
            sub.add_argument("--resume", action="store_true"); sub.add_argument("--budget-minutes", type=float)
        if name == "predict":
            sub.add_argument("--inputs", type=Path, required=True); sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--seed", type=int); sub.add_argument("--grid-json", type=Path)
    args = parser.parse_args()
    raise SystemExit(dispatch(study_root(args.name), args))


if __name__ == "__main__":
    main()
