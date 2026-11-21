"""Compact section 3.1 geometry experiments. Never starts native solvers."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "status", "report", "stop"))
    parser.add_argument("--study", default="section31_v1")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.study.replace("_", "").isalnum():
        parser.error("Use an alphanumeric/underscore study name")
    root = ROOT / "studies" / args.study
    if args.command == "init":
        from experiment import initialize
        initialize(root); print(root); return
    if args.command == "status":
        print((root / "state.json").read_text(encoding="utf-8")); return
    if args.command == "stop":
        (root / "stop.flag").write_text("User requested retained stop", encoding="utf-8"); return
    if args.command == "report":
        from experiment import setup_priority
        setup_priority()
        from plotting import make_report
        make_report(root)
        from summary_zh import write_summary
        write_summary(root); return
    import psutil
    state = __import__("json").loads((root / "state.json").read_text(encoding="utf-8"))
    if state["status"] == "COMPLETE" and (root / "report_manifest.json").exists():
        from experiment import verify, read, digest, validate_result
        verify(root); identity = digest(root / "protocol.json")
        for path in (root / "cases").glob("*/result.json"):
            validate_result(path, identity)
        for name, expected in read(root / "report_manifest.json")["artifacts"].items():
            if digest(root / name) != expected: raise RuntimeError("Completed report artifact changed")
        print("Study already complete; all saved case/report hashes validated. No worker started."); return
    if "pid" in state:
        try:
            process = psutil.Process(state["pid"])
            if abs(process.create_time() - state["created"]) < .1 and state["status"] == "RUNNING":
                raise RuntimeError("An owned controller is already running")
        except psutil.NoSuchProcess:
            pass
    if (root / "stop.flag").exists():
        if not args.resume:
            raise RuntimeError("Use --resume to continue a stopped study")
        (root / "stop.flag").unlink()
    command = [sys.executable, str(root / "source/experiment.py"), "run", "--study", str(root)]
    if args.limit:
        command += ["--limit", str(args.limit)]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       NUMEXPR_NUM_THREADS="2", VTK_SMP_MAX_THREADS="2", PYTHONIOENCODING="utf-8")
    if not args.background:
        raise SystemExit(subprocess.run(command, env=environment).returncode)
    with (root / "controller.stdout.log").open("ab") as stdout, (root / "controller.stderr.log").open("ab") as stderr:
        child = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(f"Background geometry controller PID {child.pid}")


if __name__ == "__main__":
    main()
