"""Launch one candidate on vy-nebius-2 and preserve its record (fixed; run from an agent VM's checkout of a pushed commit).

    python benchmarks/covert_witness/launch.py CLASS START [--note TEXT] [--direction D] [--mechanism M] [-- TRAIN_ARGS...]

The candidate runs as `research run --on vy-nebius-2 --pool gpu` (one slot per GPU; candidate.sh leases the GPU with gpu-lease)
from this checkout's HEAD, which must be committed. Its train.py runs inside the harness (`te_core.run --script`), which
counts its MACs and times its steps. When it ends, its `out/` (the harness's run.json, the witness record, the step log, E
when at most 1 MB, logs, the candidate's train.py) is fetched and put in the evidence store as `evidence/v1`, preserved; the
run is ingested into the te_core ledger as agent D (`python -m te_core.ledger --agent D --publish ingest`, one submission
record per checkpoint, each put in the store), and the frontier of the local ledger goes to stderr. The art id, the ledger
ids and the witness summary are printed as one JSON line, and appended to `$CW_LEDGER` (default
~/.research/covert-witness.jsonl).

D, the direction, defaults to the branch's `cursor/te-<D>-NNNN` (else `integrator`); M, the mechanism, to the note's first
word (else the class). A `reference` candidate is ingested as a reference (honest) run.
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
#: candidate.sh's GPU lease (1.2x prepare.GPU_SECONDS_CAP plus 25 minutes) and a margin
TIMEOUT = "100m"


def research(*args: str, check: bool = True, stdout_only: bool = False) -> str:
    r = subprocess.run(["research", *args], capture_output=True, text=True, cwd=REPO)
    if check and r.returncode:
        raise SystemExit(f"launch: research {' '.join(args[:2])} failed ({r.returncode}): {r.stderr[-2000:]}")
    return r.stdout if stdout_only else r.stdout + r.stderr


def tag(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", text.lower())[:64]


def branch_direction() -> str:
    b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    m = re.fullmatch(r"cursor/te-(.+)-[0-9a-f]{4}", b)
    return tag(m.group(1)) if m else "integrator"


def main() -> int:
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("klass")
    ap.add_argument("start")
    ap.add_argument("--note", default="")
    ap.add_argument("--direction", default=None, help="the ledger agent and direction (default: from the branch)")
    ap.add_argument("--mechanism", default=None, help="the mechanism tag (default: the note's first word)")
    ap.add_argument("--lane", default=os.environ.get("RESEARCH_LANE", "training"))
    ap.add_argument("--no-wait", action="store_true", help="print the run id and return; `--collect RUN` later")
    ap.add_argument("--collect", default=None, metavar="RUN", help="collect an ended candidate run")
    a = ap.parse_args(argv)
    direction = tag(a.direction) if a.direction else branch_direction()
    mechanism = tag(a.mechanism or (a.note.split() or [a.klass])[0]) or a.klass
    if a.collect:
        return collect(a.collect, a.klass, a.start, a.note, direction, mechanism)
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode:
        raise SystemExit("launch: commit your train.py first: the run ships HEAD, and its record names that commit")
    cmd = ["run", "--on", NODE, "--project", "verity", "--source", str(REPO), "--pool", "gpu", "--as", a.lane,
           "--pool-wait", "7200", "--timeout", TIMEOUT, "--declared-output", "out/*", "--",
           "bash", "benchmarks/covert_witness/candidate.sh", a.klass, a.start, *train_args]
    out = research(*cmd)
    m = re.search(r"run (r\d{8}-\d{6}-[0-9a-f]{4}) launched", out)
    if not m:
        raise SystemExit(f"launch: no run id in research run's output:\n{out[-2000:]}")
    run = m.group(1)
    print(json.dumps({"run": run, "class": a.klass, "start": a.start, "direction": direction, "mechanism": mechanism}),
          flush=True)
    if a.no_wait:
        return 0
    while True:
        research("fetch", run, check=False)
        st = json.loads((RUNS / run / "status.json").read_text()) if (RUNS / run / "status.json").exists() else {}
        if st.get("state") in DONE:
            break
        time.sleep(30)
    return collect(run, a.klass, a.start, a.note, direction, mechanism)


def ingest(run: str, out: Path, klass: str, direction: str, mechanism: str) -> dict:
    """The harness's record into the te_core ledger (each record published), then the local frontier to stderr."""
    if not (out / "run" / "run.json").exists():
        return {"ledger_error": "no run/run.json"}
    r = subprocess.run([sys.executable, "-m", "te_core.ledger", "--agent", direction, "--publish", "ingest",
                        str(out / "run"), "--direction", direction, "--mechanism", mechanism, "--run-id", run,
                        *(["--reference"] if klass == "reference" else [])], capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        return {"ledger_error": r.stderr.strip()[-500:]}
    recs = json.loads(r.stdout.strip().splitlines()[-1])
    f = subprocess.run([sys.executable, "-m", "te_core.frontier"], capture_output=True, text=True, cwd=HERE)
    print(f.stdout if f.returncode == 0 else f"launch: te_core.frontier failed: {f.stderr[-500:]}", file=sys.stderr)
    return {"ledger": [x["id"] for x in recs], "macs": recs[-1]["macs"] if recs else None,
            "rho": recs[-1]["rho"] if recs else None}


def collect(run: str, klass: str, start: str, note: str, direction: str, mechanism: str) -> int:
    research("fetch", run, "--all")
    out = RUNS / run / "out"
    w = json.loads((out / "witness.json").read_text()) if (out / "witness.json").exists() else None
    put = json.loads(research("data", "put", "--kind", "evidence/v1", "--tree", str(out), "--preserve", "--json",
                              "--meta", json.dumps({"covert_witness": {"run": run, "class": klass, "start": start,
                                                                       "note": note, "direction": direction,
                                                                       "mechanism": mechanism}}), stdout_only=True))
    row = {"run": run, "art": put["id"], "class": klass, "start": start, "note": note, "direction": direction,
           "mechanism": mechanism, "valid": w and w.get("valid"), "bpb": w and w.get("score", {}).get("bpb"),
           "E_bytes": w and w["E"]["bytes"], "phi": w and w.get("structure", {}).get("phi"),
           "lowest_updated_layer": w and w.get("structure", {}).get("lowest_updated_layer"),
           "train_flops": w and w.get("harness", {}).get("flops"), "gpu_seconds": w and w.get("gpu_seconds"),
           "checks": w and w.get("checks")}
    row |= ingest(run, out, klass, direction, mechanism)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return 0 if w else 1


if __name__ == "__main__":
    sys.exit(main())
