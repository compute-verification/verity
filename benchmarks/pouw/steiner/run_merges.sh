#!/bin/bash
# Conjecture B / Question C: exact minimum edges with merges and copies by SAT (merges_sat.py), the plans in $PLANS
# (default small,kappa), $PROCS solver processes (default: this process's cores, at most 64), $JOB_TIMEOUT seconds per
# SAT call (default 5400). Run from the clone's root (on cpu-1 through slot.py); results in $OUT/merges.jsonl.
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cd benchmarks/pouw/steiner
n=$(nproc)
procs=${PROCS:-$((n < 64 ? n : 64))}
uv venv -q "$d/venv"
VIRTUAL_ENV="$d/venv" uv pip install -q python-sat==1.9.dev15
"$d/venv/bin/python" -c "import pysat; print('pysat', pysat.__version__)" >"$OUT/env.txt"
echo "procs $procs on $(taskset -pc $$)" >>"$OUT/env.txt"
"$d/venv/bin/python" merges_sat.py --out "$OUT/merges.jsonl" batch --plan "${PLANS:-small,kappa}" \
    --timeout "${JOB_TIMEOUT:-5400}" --procs "$procs"
