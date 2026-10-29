#!/usr/bin/env bash
# The job on the B200 (sm_100a): tcgen05.mma kind::f8f6f4 and kind::mxf8f6f4 captured and scored (f8_sm100.py).
#   f8_sm100.sh [GEN ARGS]   build tcgen05_f8.cu (SASS gate: exactly one UTC*MMA per kernel), gen into a kept scratch, under
#                            one gpu-lease the capture (layout gate first), then eval on the CPU: $OUT/{capture,summary,
#                            fixture}.json, the SASS opcodes and the device identity.
#   f8_sm100.sh eval SCRATCH [EVAL ARGS]   re-score a kept scratch (no GPU).
#
# From the source tree's root on a B200:
#   OUT=DIR bash benchmarks/tc_probe_fp4/f8_sm100.sh [--tiles N] [--bulk-scale F] [--seed S]
# Exit: 0 captured and scored, 3 a kind failed the layout gate (the others captured and scored), 1 build or SASS gate failed,
# 2 no device or no nvcc.
set -uo pipefail
OUT=${OUT:?OUT names the output directory}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$(cd "$HERE/../.." && pwd)" PYTHONDONTWRITEBYTECODE=1
PY=${PYTHON:-python3}
keep() {
  command -v research >/dev/null && research keep "$1" --owner @compute-accounting --expires 2026-10-26 --priority normal \
    --regenerate "research run --on vy-dc-b200 -- bash benchmarks/tc_probe_fp4/f8_sm100.sh" \
    --why "B200 FP8 capture: operands and captured D words, re-scored by f8_sm100.sh eval" || echo "f8_sm100.sh: not kept: $1" >&2
}

if [ "${1:-}" = eval ]; then
  SCR=${2:?f8_sm100.sh eval SCRATCH}
  shift 2
  exec "$PY" "$HERE/f8_sm100.py" eval --scratch "$SCR" --out "$OUT" "$@"
fi

NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "f8_sm100.sh: no nvcc" >&2; exit 2; }
export PATH="$(dirname "$NVCC"):$PATH"
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
[ "$cap" = 10.0 ] || { echo "f8_sm100.sh: needs sm_100a (B200), the device reports '$cap'" >&2; exit 2; }
"$NVCC" --version >"$OUT/nvcc.txt" 2>&1
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
"$NVCC" -O2 -std=c++17 -Xptxas -v -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared \
  -o "$BUILD/libtcgen05_f8.so" "$HERE/tcgen05_f8.cu" >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$BUILD/libtcgen05_f8.so" >"$BUILD/lib.sass"
grep -oE "UTC[A-Z]*MMA[A-Z0-9_.]*" "$BUILD/lib.sass" | sort | uniq -c | tee "$OUT/mma-opcodes.txt"
n_mma=$(grep -cE "UTC[A-Z]*MMA" "$BUILD/lib.sass")
[ "$n_mma" = 2 ] || { echo "f8_sm100.sh: SASS gate: $n_mma UTC*MMA instructions, want one per kernel (2)" >&2; exit 1; }

base=/workspace/scratch/b200-fp8
mkdir -p "$base" 2>/dev/null || base=$HOME/scratch/b200-fp8
SCR=${F8_SCRATCH:-$base/$(basename "${RESEARCH_RUN_DIR:-$(mktemp -u f8-XXXXXX)}")}
mkdir -p "$SCR" || exit 1
keep "$SCR"
echo "$SCR" >"$OUT/scratch.txt"
"$PY" "$HERE/f8_sm100.py" gen --scratch "$SCR" "$@" || exit 1
cp "$SCR/plan.json" "$OUT/plan.json"

lease=()
command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
"${lease[@]}" bash -c 'nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
  -i "${GPU_LEASE_UUID:-0}" >"$OUT/gpu.csv" 2>&1; "$1" "$2/f8_sm100.py" capture --scratch "$3" --lib "$4" --out "$OUT"' \
  _ "$PY" "$HERE" "$SCR" "$BUILD/libtcgen05_f8.so"
rc=$?
[ $rc = 0 ] || [ $rc = 3 ] || exit $rc
[ $rc = 0 ] || echo "f8_sm100.sh: a kind failed its layout gate (capture.json kind_layout_ok); scoring the others" >&2
"$PY" "$HERE/f8_sm100.py" eval --scratch "$SCR" --out "$OUT" --procs 14 || exit 1
exit $rc
