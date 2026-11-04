"""PORTCHK: pull each run's results.jsonl and, for each failed PR, its Lean errors with context, over ssh.

    python3 /tmp/portchk/fetch.py           (runs in RUNS)

Writes /tmp/portchk/results/<run>.jsonl and /tmp/portchk/results/<run>/<N>.err (each `error:` line of the PR's log
and the 14 lines after it, at most 60 errors)."""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

RUNS = {
    "r20261103-223617-52f6": "vy-nebius-cpu-12",  # A2
    "r20261103-223644-ff12": "vy-nebius-cpu-12",  # B2
    "r20261103-222532-6526": "vy-nebius-cpu-11",  # C
    "r20261103-222541-89f3": "vy-nebius-cpu-11",  # D
    "r20261103-224800-304d": "vy-nebius-cpu-3",  # E
    "r20261103-222317-f7eb": "vy-nebius-cpu-12",  # A (cancelled; non-Security packages only)
}
RES = Path("/tmp/portchk/results")
BIN = "/agent/repos/verity/.git/verity-auth/bin:/agent/repos/verity/.venv/bin"

REMOTE = r"""
set -e
cd /workspace/research/runs/{run} 2>/dev/null || exit 0
T=$(mktemp -d)
cp out/results.jsonl $T/ 2>/dev/null || true
cp out/seeded.json $T/ 2>/dev/null || true
tail -5 stdout.log > $T/stdout.tail 2>/dev/null || true
cat status.json > $T/status.json 2>/dev/null || true
mkdir -p $T/err
for f in out/logs/*.log; do
  [ -f "$f" ] || continue
  n=$(basename $f .log)
  grep -n -A14 -E "^error" "$f" | head -900 > $T/err/$n.err || true
  grep -E "^lean-(build|changed|records)" "$f" | cut -c1-600 > $T/err/$n.steps || true
done
tar czf - -C $T . | base64 -w0
rm -rf $T
"""


def fetch(run: str, node: str) -> None:
    env = {**os.environ, "PATH": BIN + ":" + os.environ["PATH"]}
    p = subprocess.run(["research", "pods", "ssh", node, "--", REMOTE.format(run=run)], capture_output=True, text=True,
                       env=env, timeout=180)
    data = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    if not data:
        print(run, "nothing", p.stderr[-300:])
        return
    d = RES / run
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(data)), mode="r:gz") as t:
        t.extractall(d, filter="data")
    n = sum(1 for _ in open(d / "results.jsonl")) if (d / "results.jsonl").exists() else 0
    print(run, node, f"{n} results;", (d / "stdout.tail").read_text().strip().splitlines()[-1:] if (d / "stdout.tail").exists() else "")


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    for run, node in RUNS.items():
        try:
            fetch(run, node)
        except Exception as e:
            print(run, "failed:", e)
