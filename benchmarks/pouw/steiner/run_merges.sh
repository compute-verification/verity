#!/bin/bash
# Conjecture B / Question C: exact minimum edges with merges and copies by SAT (merges_sat.py), the plans in $PLANS
# (default w4), $PROCS solver processes (default: this process's cores, at most 64), $JOB_TIMEOUT seconds per
# SAT call (default 3600). Run from the clone's root (on cpu-1 through slot.py); results in $OUT/merges.jsonl, written
# as each call ends, then every SAT model replayed by merges_verify.py and lifted (verify.txt, lift.txt).
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cd benchmarks/pouw/steiner
n=$(nproc)
procs=${PROCS:-$((n < 64 ? n : 64))}
uv venv -q "$d/venv"
VIRTUAL_ENV="$d/venv" uv pip install -q python-sat==1.9.dev15
"$d/venv/bin/python" -c "import sys, pysat; print('python', sys.version.split()[0], 'pysat', pysat.__version__)" >"$OUT/env.txt"
echo "procs $procs on $(taskset -pc $$)" >>"$OUT/env.txt"
"$d/venv/bin/python" merges_sat.py --out "$OUT/merges.jsonl" batch --plan "${PLANS:-w4}" \
    --timeout "${JOB_TIMEOUT:-3600}" --procs "$procs"
"$d/venv/bin/python" merges_verify.py check "$OUT/merges.jsonl" >"$OUT/verify.txt"
"$d/venv/bin/python" merges_verify.py lift "$OUT/merges.jsonl" --m 2,3,4 >"$OUT/lift.txt"
if grep -q '"ok": false' "$OUT/verify.txt" "$OUT/lift.txt"; then
    echo "merges_verify: a model failed its replay" >&2
    exit 1
fi
