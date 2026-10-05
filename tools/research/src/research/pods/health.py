"""`research pods health <pod> [--json] [--no-bench] [--seconds S] [--record]`: is the pod fit to measure on?

Runs `backends/direct/ligero/pod_health.py` on the pod, over ssh, in the synced tree with the bootstrap's env.sh: the cgroup CPU
quota vs nproc, GPU clocks / power limit / PCIe / persistence, other GPU processes, disk, and a ~20 s encode + matmul
microbenchmark against the SKU's entry in `backends/direct/ligero/pod_health_ref.json`.  Exit 3 = DEGRADED: replace the pod.

`--record` folds this pod's numbers into the worktree's `pod_health_ref.json` (per metric, the entry stays unless this pod beats
it: the reference is the best pod seen); commit the file.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import remote
from . import connect

REF = Path("backends/direct/ligero/pod_health_ref.json")
BETTER = {"encode_ms": min, "matmul_tflops": max}


def merge(refs: dict[str, Any], rep: dict[str, Any], pod: str, utc: str) -> list[str]:
    """Fold a pod_health report's bench into `refs` (the reference file's document); the metrics this pod set."""
    bench = rep.get("bench") or {}
    entry = refs.setdefault("skus", {}).setdefault(rep.get("sku") or "unknown", {})
    took = []
    for metric, better in BETTER.items():
        v, old = bench.get(metric), (entry.get(metric) or {}).get("value")
        if v is not None and (old is None or better(v, old) != old):
            entry[metric] = {"value": v, "pod": pod, "utc": utc}
            took.append(metric)
    return took


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research pods health", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", metavar="TARGET", help="machine name, pod id or pod name")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-bench", action="store_true", help="facts only (no GPU work)")
    ap.add_argument("--seconds", type=float, default=None, help="timed seconds per benchmark (pod_health's default: 10)")
    ap.add_argument("--record", action="store_true", help="fold this pod's numbers into the worktree's pod_health_ref.json")
    ap.add_argument("--src", default=connect.DEFAULT_DEST, help=f"the synced tree on the pod (default {connect.DEFAULT_DEST})")
    ns = ap.parse_args(argv)
    if ns.record and ns.no_bench:
        ap.error("--record needs the benchmark")
    try:
        base = connect.ssh_base(ns.target)
    except remote.Refused as e:
        print(f"research pods health: {e}", file=sys.stderr)
        return 1
    flags = ["--json"] * (ns.json or ns.record) + ["--no-bench"] * ns.no_bench + (["--seconds", str(ns.seconds)] if ns.seconds else [])
    script = (f'. /workspace/env.sh 2>/dev/null; cd {shlex.quote(ns.src)} && "${{PY:-python3}}" -m backends.direct.ligero.pod_health '
              + shlex.join(flags))
    if not ns.record:
        return subprocess.run([*base, script], check=False).returncode
    r = subprocess.run([*base, script], capture_output=True, text=True, check=False)
    try:
        rep = json.loads(r.stdout)
    except ValueError:
        print(f"research pods health: no report from {ns.target} (rc {r.returncode}):\n{r.stdout[-2000:]}{r.stderr[-2000:]}", file=sys.stderr)
        return 1
    path = connect.git_toplevel(Path.cwd()) / REF
    refs = json.loads(path.read_text()) if path.is_file() else {"schema": "pod-health-ref/v1", "skus": {}}
    took = merge(refs, rep, ns.target, time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()))
    path.write_text(json.dumps(refs, indent=1) + "\n")
    print(f"{ns.target}: {rep.get('verdict')} {json.dumps(rep.get('bench'))}; {rep.get('sku')} reference: "
          f"{'set ' + ', '.join(took) if took else 'kept (it is better)'} in {path}")
    return 3 if rep.get("verdict") == "DEGRADED" else 0
