"""Launch one candidate on vy-nebius-2 and preserve its record (fixed; run from an agent VM's checkout of a pushed commit).

    python benchmarks/covert_witness/launch.py CLASS START [--note TEXT] [-- TRAIN_ARGS...]

The candidate runs as `research run --on vy-nebius-2 --pool gpu` (one slot per GPU; candidate.sh leases the GPU with gpu-lease)
from this checkout's HEAD, which must be committed. When it ends, its `out/` (witness record, step log, E when at most 1 MB,
logs, the candidate's train.py) is fetched and put in the evidence store as `evidence/v1`, preserved; the art id and the
witness summary are printed as one JSON line, and appended to `$CW_LEDGER` (default ~/.research/covert-witness.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NODE = "vy-nebius-2"
RUNS = Path.home() / ".research" / "runs"
LEDGER = Path(os.environ.get("CW_LEDGER", Path.home() / ".research" / "covert-witness.jsonl"))
DONE = {"done", "failed", "cancelled", "refused", "timeout", "lost"}


def research(*args: str, check: bool = True, stdout_only: bool = False) -> str:
    r = subprocess.run(["research", *args], capture_output=True, text=True, cwd=REPO)
    if check and r.returncode:
        raise SystemExit(f"launch: research {' '.join(args[:2])} failed ({r.returncode}): {r.stderr[-2000:]}")
    return r.stdout if stdout_only else r.stdout + r.stderr


def main() -> int:
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("klass")
    ap.add_argument("start")
    ap.add_argument("--note", default="")
    ap.add_argument("--lane", default=os.environ.get("RESEARCH_LANE", "training"))
    ap.add_argument("--no-wait", action="store_true", help="print the run id and return; `--collect RUN` later")
    ap.add_argument("--collect", default=None, metavar="RUN", help="collect an ended candidate run")
    a = ap.parse_args(argv)
    if a.collect:
        return collect(a.collect, a.klass, a.start, a.note)
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode:
        raise SystemExit("launch: commit your train.py first: the run ships HEAD, and its record names that commit")
    cmd = ["run", "--on", NODE, "--project", "verity", "--source", str(REPO), "--pool", "gpu", "--as", a.lane,
           "--pool-wait", "7200", "--timeout", "45m", "--declared-output", "out/*", "--",
           "bash", "benchmarks/covert_witness/candidate.sh", a.klass, a.start, *train_args]
    out = research(*cmd)
    m = re.search(r"run (r\d{8}-\d{6}-[0-9a-f]{4}) launched", out)
    if not m:
        raise SystemExit(f"launch: no run id in research run's output:\n{out[-2000:]}")
    run = m.group(1)
    print(json.dumps({"run": run, "class": a.klass, "start": a.start}), flush=True)
    if a.no_wait:
        return 0
    while True:
        research("fetch", run, check=False)
        st = json.loads((RUNS / run / "status.json").read_text()) if (RUNS / run / "status.json").exists() else {}
        if st.get("state") in DONE:
            break
        time.sleep(30)
    return collect(run, a.klass, a.start, a.note)


def collect(run: str, klass: str, start: str, note: str) -> int:
    research("fetch", run, "--all")
    out = RUNS / run / "out"
    w = json.loads((out / "witness.json").read_text()) if (out / "witness.json").exists() else None
    put = json.loads(research("data", "put", "--kind", "evidence/v1", "--tree", str(out), "--preserve", "--json",
                              "--meta", json.dumps({"covert_witness": {"run": run, "class": klass, "start": start,
                                                                       "note": note}}), stdout_only=True))
    art = put["id"]
    row = {"run": run, "art": art, "class": klass, "start": start, "note": note,
           "valid": w and w.get("valid"), "bpb": w and w.get("score", {}).get("bpb"),
           "E_bytes": w and w["E"]["bytes"], "phi": w and w.get("structure", {}).get("phi"),
           "lowest_updated_layer": w and w.get("structure", {}).get("lowest_updated_layer"),
           "train_flops": w and w.get("train", {}).get("flops"), "gpu_seconds": w and w.get("gpu_seconds"),
           "checks": w and w.get("checks")}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return 0 if w else 1


if __name__ == "__main__":
    sys.exit(main())
