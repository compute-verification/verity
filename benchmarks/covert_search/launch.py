"""Run one candidate: the checkout's committed train.py for ISLAND, metered on vy-nebius-2's GPU, its witness put in the evidence
store, and a row appended to the island's log (islands/<ISLAND>/results.tsv, untracked).

    python benchmarks/covert_search/launch.py ISLAND --start m50 [--seed 0] [--hp KEY=VALUE ...] [--note TEXT]
    python benchmarks/covert_search/launch.py ISLAND --start m50 --collect RUN   # a run whose launcher died

The candidate is the commit at HEAD (research run ships it; a dirty tree is refused, so commit train.py first). Steps:
`research run --on vy-nebius-2 --pool research -- bash benchmarks/covert_search/job.sh ...` (job.sh takes one GPU with
gpu-lease; the node refuses `research run --gpus`), wait for it, `research
fetch --all`, `research data put --kind evidence/v1 --tree <run>/out --preserve`, then the log row. Prints the row as JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NODE = "vy-nebius-2"
CLASS_OF = {"reference": "ref", "mb10": "10MB", "mb1": "1MB", "kb100": "100kB"}
COLUMNS = ["utc", "commit", "train_py", "island", "class", "start", "seed", "hp", "status", "start_bpb", "final_bpb",
           "charged_flops", "gpu_s", "run", "art", "note"]


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=os.environ | {"RESEARCH_LANE": os.environ.get(
        "RESEARCH_LANE", "training")})
    if check and r.returncode:
        raise SystemExit(f"launch: {' '.join(cmd[:4])} ... exited {r.returncode}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("island", choices=sorted(CLASS_OF))
    ap.add_argument("--start", required=True, help="a registered start: s50, s100, m50, m100, smol")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--note", default="")
    ap.add_argument("--timeout", default="30m")
    ap.add_argument("--meter-arg", action="append", default=[], help="passed to meter.py (smoke tests: --budget-blocks 20)")
    ap.add_argument("--collect", metavar="RUN", help="don't run: fetch, store and log RUN, a finished candidate's run")
    a = ap.parse_args()
    cls = CLASS_OF[a.island]
    if a.collect:
        run = a.collect
    else:
        if sh(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.strip():
            raise SystemExit("launch: the checkout has uncommitted changes; commit the candidate's train.py first")
        extra = [x for kv in a.hp for x in ("--hp", kv)] + a.meter_arg
        r = sh(["research", "run", "--on", NODE, "--pool", "research", "--project", "verity", "--source", str(REPO),
                "--cwd", "source", "--timeout", a.timeout, "--wait", "--",
                "bash", "benchmarks/covert_search/job.sh", a.island, a.start, cls, str(a.seed), *extra], check=False)
        m = re.search(r"r\d{8}-\d{6}-[0-9a-f]{4}", r.stdout + r.stderr)
        if not m:
            raise SystemExit(f"launch: research run gave no run id (rc {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        run = m.group(0)
        while r.returncode == 79:  # --wait's cap; the run goes on
            time.sleep(30)
            r = sh(["research", "status", run, "--on", NODE, "--refresh"], check=False)
            r.returncode = 79 if re.search(r"\b(running|waiting|submitted|launching)\b", r.stdout) else 0
    for _ in range(6):  # exit 4 right after "done": the node's sampler still writes resources.jsonl as tar reads it
        f = sh(["research", "fetch", run, "--on", NODE, "--all"], check=False)
        if f.returncode != 4:
            break
        time.sleep(20)
    if f.returncode:
        raise SystemExit(f"launch: research fetch {run} exited {f.returncode}:\n{f.stdout[-2000:]}\n{f.stderr[-2000:]}")
    out = Path.home() / ".research/runs" / run / "out"
    w = json.loads((out / "witness.json").read_text()) if (out / "witness.json").exists() else {"status": "no witness"}
    commit = w.get("commit") or sh(["git", "rev-parse", "HEAD"]).stdout.strip()
    tp = Path.home() / ".research/runs" / run / "source" / "benchmarks/covert_search/islands" / a.island / "train.py"
    train_py = w.get("train_py_sha256") or hashlib.sha256(
        (tp if tp.exists() else HERE / "islands" / a.island / "train.py").read_bytes()).hexdigest()
    art = ""
    if (out / "witness.json").exists():
        meta = {"campaign": "training", "lane": "training", "what": "covert-search candidate witness", "run": run,
                "island": a.island, "class": cls, "start": a.start, "commit": commit}
        p = sh(["research", "data", "put", "--kind", "evidence/v1", "--tree", str(out), "--preserve", "--json",
                "--meta", json.dumps(meta)], check=False)
        am = re.search(r"art:[0-9a-f]{16,}", p.stdout) or re.search(r'"(?:art|id)"\s*:\s*"([0-9a-f]{16,})"', p.stdout)
        art = (am.group(0) if am and am.group(0).startswith("art:") else f"art:{am.group(1)}") if am else f"put failed: {p.stderr[-200:]}"
    traj = w.get("trajectory") or [{}]
    row = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "commit": commit[:12], "train_py": train_py[:12],
           "island": a.island, "class": cls, "start": w.get("start", a.start), "seed": w.get("seed", a.seed), "hp": json.dumps(w.get("hp", {}), sort_keys=True),
           "status": w.get("status") + (f": {w['refusal']}" if w.get("refusal") else ""), "start_bpb": traj[0].get("bpb"),
           "final_bpb": traj[-1].get("bpb") if w.get("status") == "complete" else None,
           "charged_flops": w.get("charged_flops"), "gpu_s": w.get("gpu_seconds"), "run": run, "art": art, "note": a.note}
    log = HERE / "islands" / a.island / "results.tsv"
    new = not log.exists()
    with log.open("a") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join("" if row[c] is None else str(row[c]) for c in COLUMNS) + "\n")
    print(json.dumps(row))
    return 0 if w.get("status") == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
