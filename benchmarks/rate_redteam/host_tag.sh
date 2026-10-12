#!/usr/bin/env bash
# The host relay's tag rate on the CPUs the run is given (cwd = the source tree; no GPU). Finds a Python with the
# `blake3` package, gates it against the plain-Python reference, then benches. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/rate_redteam/host_tag.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
D=benchmarks/rate_redteam
PY=
for c in python3 /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python; do
  if command -v "$c" >/dev/null && "$c" -c 'import blake3' 2>/dev/null; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  T=$(mktemp -d /tmp/rate-redteam-py.XXXXXX); trap 'rm -rf "$T"' EXIT
  python3 -m pip install -q --target "$T" blake3 > "$O/pip.log" 2>&1 || { tail -20 "$O/pip.log"; exit 6; }
  export PYTHONPATH="$T"; PY=python3
fi
lscpu > "$O/lscpu.txt"
"$PY" $D/host_tag.py gate > "$O/gate.log" 2>&1 || { cat "$O/gate.log"; exit 2; }
"$PY" $D/host_tag.py bench --procs 1 12 24 48 --out "$O/host-tag.json" > "$O/bench.log" 2>&1 || exit 3
cat "$O/gate.log" "$O/bench.log"
