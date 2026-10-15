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
for c in python3 /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python \
         /workspace/*/.venv/bin/python /workspace/*/*/.venv/bin/python "$HOME"/*/.venv/bin/python; do
  if [ -x "$(command -v "$c")" ] && "$c" -c 'import blake3' 2>/dev/null; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  UV=$(command -v uv || ls "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" 2>/dev/null | head -1)
  [ -n "$UV" ] || { echo "no Python with blake3 and no uv"; exit 6; }
  T=$(mktemp -d /tmp/rate-redteam-py.XXXXXX); trap 'rm -rf "$T"' EXIT
  { "$UV" venv -q "$T/v" && "$UV" pip install -q --python "$T/v/bin/python" blake3; } > "$O/uv.log" 2>&1 \
    || { tail -20 "$O/uv.log"; exit 6; }
  PY=$T/v/bin/python
fi
echo "python: $PY" > "$O/python.txt"
lscpu > "$O/lscpu.txt"
"$PY" $D/host_tag.py gate > "$O/gate.log" 2>&1 || { cat "$O/gate.log"; exit 2; }
"$PY" $D/host_tag.py bench --procs 1 12 24 48 --out "$O/host-tag.json" > "$O/bench.log" 2>&1 || exit 3
cat "$O/gate.log" "$O/bench.log"
