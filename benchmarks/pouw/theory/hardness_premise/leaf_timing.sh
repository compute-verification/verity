#!/usr/bin/env bash
# The correction's price on one GPU (leaf_corr.cu): the fixture (leaf_fixture.py), the build and its SASS, then the gated
# timing, under research run with $OUT. One GPU, about two minutes of it; the build and the fixture are seconds.
#
#   research run --on vy-nebius-1 --gpus 1 --source TREE --cwd source -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots \
#     python3 tools/verity/check/slot.py --by LANE -- bash benchmarks/pouw/theory/hardness_premise/leaf_timing.sh
#
# ARCH (sm_120a; sm_90a on an H200), PEAK (the W1 rate, s8 MAC an SM-clock; unset: the card's mma.sync rate, measured)
# and CORRS (the corrections to gate and time, leaf_corr's --corrs; unset: all eight) come from the environment
# (`research run --env`).
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARCH=${ARCH:-sm_120a}
NVCC=${NVCC:-}
for c in /usr/local/cuda-13.0/bin/nvcc /usr/local/cuda/bin/nvcc "$(command -v nvcc)"; do
  [ -z "$NVCC" ] && [ -x "$c" ] && NVCC=$c
done
[ -n "$NVCC" ] || { echo "no nvcc" >&2; exit 1; }
PY=python3
$PY -c "import numpy" 2>/dev/null || PY="uv run --no-project --with numpy python3"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
{
  echo "run ${RESEARCH_RUN_ID:-?} host $(hostname) start $(date -u +%FT%TZ) arch $ARCH peak ${PEAK:-measured}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} GPU_LEASE_UUID=${GPU_LEASE_UUID:-}"
  "$NVCC" --version | tail -2
  nvidia-smi --query-gpu=index,uuid,name,compute_cap,clocks.max.sm,power.limit --format=csv
} >"$OUT/meta.txt" 2>&1
$PY "$SRC/leaf_fixture.py" "$WORK/fx" >"$OUT/fixture.log" 2>&1 || { tail -20 "$OUT/fixture.log" >&2; exit 1; }
cp "$WORK/fx/fixture.json" "$OUT/"
"$NVCC" -O3 -std=c++17 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" -Xptxas -v -o "$WORK/leaf_corr" "$SRC/leaf_corr.cu" \
  >"$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log" >&2; exit 1; }
"$(dirname "$NVCC")/cuobjdump" -sass "$WORK/leaf_corr" >"$OUT/leaf_corr.sass" 2>&1 || true
if grep -qE "[1-9][0-9]* bytes spill" "$OUT/build.log"; then
  echo "a kernel spills (build.log): no timing" >&2
  exit 1
fi
SMI=
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  nvidia-smi --id="${GPU_LEASE_UUID%%,*}" --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu \
    --format=csv,noheader -lms 200 >"$OUT/smi.csv" 2>&1 &
  SMI=$!
fi
timeout 900 "$WORK/leaf_corr" "$WORK/fx" "$OUT" ${PEAK:+--peak "$PEAK"} ${CORRS:+--corrs "$CORRS"} \
  >"$OUT/leaf_corr.jsonl" 2>"$OUT/leaf_corr.err"
rc=$?
[ -n "$SMI" ] && kill "$SMI" 2>/dev/null
cat "$OUT/leaf_corr.jsonl"
tail -5 "$OUT/leaf_corr.err" >&2
exit $rc
