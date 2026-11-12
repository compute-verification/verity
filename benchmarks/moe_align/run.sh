#!/usr/bin/env bash
# The canonical moe_align_block_size's checks and timing on one GPU, in a venv of the integration's pinned vLLM wheel (the pin
# read from pod_bootstrap.sh; the venv built when it lacks vLLM): stress.py check in one process, then in two at once; their
# comparison (stress.py compare: summary.json); the integration's plugin as vLLM loads it, from a site directory declaring
# the integration's entry points (stress.py site, then stress.py plugin: plugin.json); the timing
# (stress.py time: time.json). A failed step is named in $OUT/failed and the script exits 1 after the rest.
#   research run --on NODE --project verity --source . --cwd source -- bash benchmarks/moe_align/run.sh
# VENV (default /workspace/venv312), PY (default $VENV/bin/python), OUT (research run sets it); MOE_ALIGN_REPS: the first
# check's repetitions (default 100; the two concurrent ones run 10 each); MOE_ALIGN_SKIP_TIME=1: no timing.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${OUT:?set OUT}
VENV=${VENV:-/workspace/venv312}
PY=${PY:-$VENV/bin/python}
mkdir -p "$OUT"
if ! "$PY" -c "import vllm, torch, triton" 2>/dev/null; then
  BOOT=$ROOT/adapters/vllm/verity_vllm/ops/pod_bootstrap.sh
  eval "$(grep -E '^(W|WSHA|TORCHCODEC)=' "$BOOT")"
  URL=$(grep -o 'https://wheels.vllm.ai/[^"]*' "$BOOT")
  command -v uv > /dev/null || { python3 -m pip install -q uv; export PATH=$HOME/.local/bin:$PATH; }
  [ -x "$PY" ] || uv venv "$VENV" --python 3.12 > /dev/null || exit 3
  wheels=$(mktemp -d)
  curl -fsSLo "$wheels/$W" "$URL" && echo "$WSHA  $wheels/$W" | sha256sum -c > /dev/null || exit 3
  UV_HTTP_TIMEOUT=600 uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match "$wheels/$W" "$TORCHCODEC" "numpy>=1.26" > "$OUT/pip.log" 2>&1 || exit 3
  rm -rf "$wheels"
fi
export PYTHONPATH=$ROOT:$ROOT/adapters/vllm
run() { local name=$1; shift; "$@" > "$OUT/$name.log" 2>&1 || echo "$name" >> "$OUT/failed"; }
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv > "$OUT/gpu.csv"
"$PY" -c "import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)" > "$OUT/versions.txt"
run check.p0 "$PY" "$HERE/stress.py" check --out "$OUT" --tag p0 --reps "${MOE_ALIGN_REPS:-100}"
run check.p1 "$PY" "$HERE/stress.py" check --out "$OUT" --tag p1 --reps 10 & a=$!
run check.p2 "$PY" "$HERE/stress.py" check --out "$OUT" --tag p2 --reps 10 & b=$!
wait "$a" "$b"
run compare "$PY" "$HERE/stress.py" compare --out "$OUT"
site=$(mktemp -d)
"$PY" "$HERE/stress.py" site --out "$site" > /dev/null
PYTHONPATH="$site:$PYTHONPATH" run plugin "$PY" "$HERE/stress.py" plugin --out "$OUT" --reps 3
rm -rf "$site"
[ "${MOE_ALIGN_SKIP_TIME:-0}" = 1 ] || run time "$PY" "$HERE/stress.py" time --out "$OUT"
head -12 "$OUT/compare.log"
[ ! -s "$OUT/failed" ] || { echo "failed: $(tr '\n' ' ' < "$OUT/failed")"; exit 1; }
