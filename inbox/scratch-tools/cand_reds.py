#!/usr/bin/env python3
"""Every candidate's shard runs on the check nodes, with what has failed in them so far.

usage: cand_reds.py [RUN_PREFIX]   (default: runs of the last 3 hours, by run id)

For each run on a check node whose source is a commit: node, run id, source sha (12), shard, age in minutes,
each step's state and seconds, and every failed suite from check-live.json's `progress.failed` and
`suites.failed.json`. A step stays "running" after a suite in it fails, so the failed list is the thing to read.
Run from /tmp/rchk with PYTHONPATH=/tmp/rchk/tools/research/src RESEARCH_AT_CHECKOUT=1.
"""
import datetime
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

NODES = [f"vy-nebius-cpu-{n}" for n in (1, 2, 4, 5, 6, 7, 9, 11, 13)]

REMOTE = r'''
import glob, json, os, re, sys, time
since = sys.argv[1]
out = []
for d in sorted(glob.glob("/workspace/research/runs/r2026*")):
    rid = os.path.basename(d)
    if rid < since:
        continue
    try:
        launch = open(d + "/launch.json").read()
    except OSError:
        continue
    m = re.search(r'source_sha\\": \\"([0-9a-f]{12})', launch)
    sh = re.search(r'--shard[" ,\\n]+([0-9]+/[0-9]+)', launch)
    row = {"run": rid, "sha": m.group(1) if m else "?", "shard": sh.group(1) if sh else "-"}
    try:
        st = json.load(open(d + "/status.json"))
        row["state"] = st.get("state") or st.get("status")
        row["rc"] = st.get("rc", st.get("returncode"))
    except (OSError, ValueError):
        row["state"] = "?"
    try:
        live = json.load(open(d + "/check-live.json"))
    except (OSError, ValueError):
        live = None
    if live is None and row["shard"] == "-":
        continue
    steps = []
    failed = []
    if live:
        seq = live.get("steps") or []
        if isinstance(seq, dict):
            seq = [dict(v, name=k) for k, v in seq.items()]
        for s in seq:
            name = s.get("name") or s.get("step")
            steps.append([name, s.get("state") or s.get("status"), s.get("seconds") or s.get("elapsed")])
            prog = s.get("progress") or {}
            for f in prog.get("failed") or []:
                failed.append([name, f if isinstance(f, str) else json.dumps(f)[:200]])
    try:
        sf = json.load(open(d + "/suites.failed.json"))
        row["suites_failed"] = sf if isinstance(sf, list) else list(sf)
    except (OSError, ValueError):
        row["suites_failed"] = []
    row["steps"] = steps
    row["failed"] = failed
    row["mtime"] = int(time.time() - os.path.getmtime(d + "/launch.json"))
    out.append(row)
load = open("/proc/loadavg").read().split()[0]
print(json.dumps({"load": load, "cores": os.cpu_count(), "runs": out}))
'''


def read(node, since):
    cmd = ["python3", "-m", "research", "pods", "ssh", node, "--", "python3", "-", since]
    try:
        p = subprocess.run(cmd, input=REMOTE, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return node, None, "timeout"
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return node, json.loads(line), ""
            except ValueError:
                pass
    return node, None, (p.stderr or p.stdout)[-300:]


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    since = sys.argv[1] if len(sys.argv) > 1 else (now - datetime.timedelta(hours=3)).strftime("r%Y%m%d-%H%M%S")
    print(f"read {now:%H:%M:%SZ}, runs from {since}")
    with ThreadPoolExecutor(len(NODES)) as ex:
        results = list(ex.map(lambda n: read(n, since), NODES))
    by_sha = {}
    for node, data, err in results:
        if data is None:
            print(f"{node}: NOT READ: {err.strip()[:200]}")
            continue
        print(f"{node}: load {data['load']} of {data['cores']}, {len(data['runs'])} runs")
        for r in data["runs"]:
            by_sha.setdefault(r["sha"], []).append((node, r))
    for sha, rows in sorted(by_sha.items(), key=lambda kv: min(r["run"] for _, r in kv[1])):
        print(f"\n== {sha}")
        for node, r in sorted(rows, key=lambda nr: (nr[1]["shard"], nr[1]["run"])):
            steps = ", ".join(f"{n}:{s}" + (f" {int(sec)}s" if isinstance(sec, (int, float)) else "") for n, s, sec in r["steps"] if s not in ("pending", "skipped", None))
            print(f"  {r['shard']:>4} {node[10:]:>6} {r['run']} {r['state']} rc={r.get('rc')} launched {r['mtime'] // 60} min ago")
            if steps:
                print(f"       {steps[:600]}")
            # circuit-check's progress counts HELD rows (block_held.HELD, 121 entries) as failures; its report does not,
            # and the step is off the gate (policy.toml). Only the step's own state says whether it failed.
            cc = [f for f in r["failed"] if f[0] == "circuit-check"]
            gate = [f for f in r["failed"] if f[0] != "circuit-check"]
            if gate or r["suites_failed"]:
                print(f"       RED: progress.failed={gate} suites.failed={r['suites_failed']}")
            if cc:
                print(f"       circuit-check (off gate): {len(cc)} rows held or failed in progress; read the step's state, not this")


if __name__ == "__main__":
    main()
