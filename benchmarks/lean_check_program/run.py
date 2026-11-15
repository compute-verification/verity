"""What `tools/verity/lean/check.py` spends, call by call, on getting its own Lean programs (`Facts.lean`, `Unread.lean`)
against what the rest of a call costs, and its guard.

    research run --on auto --cwd source -- \
      bash -lc 'uv run python3 benchmarks/lean_check_program/run.py control --out "$OUT" [--calls N] [--cache DIR]'

PACKAGE is a package directory with a `lean-audit.json`, or `control`, the auditor's control package copied and built in a
scratch directory. Each call is `check.py --no-replay --no-runs` on it in a fresh process, as the graph's lean-lock runner
and the tests start one, with the time of each stage: the process's start until `main`, `program` (getting a program, by
compiling it or from the program cache), running each program, and the rest. ``--cache DIR`` hands every call the program
cache at DIR (`check.py --program-cache`), which the first call fills. The guard holds when every call reports the same
failures and records. It writes `check-program.json` under ``--out`` and exits 1 when the guard fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "tools" / "verity" / "lean" / "check.py"
CONTROL = ROOT / "tools" / "verity" / "lean" / "tests" / "fixtures" / "control"
#: what of a package's report the guard compares: everything but how long it took and where it ran
TIMING = ("seconds", "peak_gib", "package", "regenerated", "facts_document", "work")


def child(report: Path, argv: list[str]) -> int:
    """One `check.py` call in this process, timed by stage, its stages written to ``report``."""
    t0 = time.monotonic()
    spec = importlib.util.spec_from_file_location("check", CHECK)
    check = importlib.util.module_from_spec(spec)
    sys.modules["check"] = check
    spec.loader.exec_module(check)
    seconds: dict[str, float] = defaultdict(float)
    seconds["start"] = time.monotonic() - t0
    program, run = check.program, check.run

    def timed_program(*a, **k):
        s = time.monotonic()
        try:
            return program(*a, **k)
        finally:
            seconds[f"program {a[1]}"] += time.monotonic() - s

    def timed_run(argv, *a, **k):
        s = time.monotonic()
        try:
            return run(argv, *a, **k)
        finally:
            name = next((n for x in argv for n in ("facts", "unread") if Path(x).name.split("-")[0] == n), None)
            if name and "-o" not in argv:
                seconds[f"run {name}"] += time.monotonic() - s

    check.program, check.run = timed_program, timed_run
    s = time.monotonic()
    rc = check.main(argv)
    seconds["main"] = time.monotonic() - s
    report.write_text(json.dumps({"rc": rc, "seconds": {k: round(v, 2) for k, v in seconds.items()}}))
    return 0


def built_control(scratch: Path) -> Path:
    pkg = scratch / "control"
    shutil.copytree(CONTROL, pkg, ignore=shutil.ignore_patterns(".lake"))
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True)
    return pkg


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(Path(sys.argv[2]), sys.argv[3:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("package", help="a package directory, or `control`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--cache", type=Path, help="the program cache every call is handed (`check.py --program-cache`)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lean-check-program-") as d:
        scratch = Path(d)
        pkg = built_control(scratch) if a.package == "control" else Path(a.package).resolve()
        calls, reports = [], []
        for i in range(a.calls):
            out, stages = scratch / f"out-{i}", scratch / f"stages-{i}.json"
            argv = ["--no-replay", "--no-runs", "--out", str(out), str(pkg)]
            argv += ["--program-cache", str(a.cache)] if a.cache else []
            s = time.monotonic()
            subprocess.run([sys.executable, __file__, "--child", str(stages), *argv], check=True)
            wall = time.monotonic() - s
            got = json.loads(stages.read_text())
            doc = json.loads((out / "audit.json").read_text())
            rep = doc["packages"][0]
            got["seconds"] = {"wall": round(wall, 2), **got["seconds"], **{f"report {k}": v for k, v in
                                                                            rep.get("seconds", {}).items()}}
            calls.append(got)
            reports.append({k: v for k, v in rep.items() if k not in TIMING})
            print(f"call {i}: rc {got['rc']}, " + ", ".join(f"{k} {v}" for k, v in got["seconds"].items()), flush=True)
    same = all(r == reports[0] for r in reports) and all(c["rc"] == calls[0]["rc"] for c in calls)
    doc = {"package": a.package, "cache": str(a.cache) if a.cache else None, "calls": calls, "same": same}
    (a.out / "check-program.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"guard: {'every call reports the same' if same else 'the calls report differently'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
