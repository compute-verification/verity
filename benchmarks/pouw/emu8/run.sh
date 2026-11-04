#!/usr/bin/env bash
# EMU8's job: build and SASS-gate outside the lease; under one gpu-lease the exactness gate (every prefix word of every
# arm, poisoned, with its no-write control) and then the timed shapes (each gated again on its final words first); the
# host's recheck of the dumped words after the lease.
#   research run --on vy-nebius-2 --project verity --cwd source --env GPU_LEASE_WHO=LANE -- \
#     bash benchmarks/pouw/emu8/run.sh [REPS]
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KIT=$HERE/../gpu_kit
REPS=${1:-10}
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}; [ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
export LD_LIBRARY_PATH="$(dirname "$NVCC")/../lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
# Llama-3.1-8B at M = 4096 tokens: qkv, o_proj, gate/up (fused), down
SHAPES=(4096 6144 4096  4096 4096 4096  4096 28672 4096  4096 4096 14336)

"$NVCC" -std=c++17 -O3 -Xptxas -v -gencode arch=compute_120a,code=sm_120a -o "$WORK/emu8" "$HERE/emu8.cu" -lcublasLt \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/emu8" >"$OUT/emu8.sass"
# The timed mainloop a kernel: imma 32 IMMAs a 2-step stage; k3 48 QMMAs and k4 64 a step (gemm<ARM, FOLD, DUMP>)
python3 "$HERE/emu8.py" identities >"$OUT/identities.json" || exit 9
rc=0
for spec in "0:IMMA*=32" "1:QMMA*=48" "2:QMMA*=64"; do
  arm=${spec%%:*} expect=${spec#*:}
  for fold in 0 1; do
    python3 "$KIT/sass_gate.py" "$WORK/emu8" --kernels "_Z4gemmILi${arm}ELb${fold}ELb0E*" --expect "$expect" --numeric \
      --json "$OUT/sass-gate-$arm-$fold.json" >>"$OUT/sass-gate.txt" 2>&1 || rc=8
  done
done
cat "$OUT/sass-gate.txt"; [ $rc -eq 0 ] || exit $rc

gpu-lease 1 --wait --max-min 25 -- bash -c '
  nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv >"$3/gpu.csv"
  "$1/emu8" gate "$1/gate" >"$3/gate.jsonl"; rc=$?
  [ $rc -eq 0 ] || exit $rc
  "$1/emu8" time "$2" "${@:4}" >"$3/time.jsonl"' _ "$WORK" "$REPS" "$OUT" "${SHAPES[@]}"
rc=$?
tail -n 2 "$OUT/gate.jsonl" 2>/dev/null
python3 "$HERE/emu8.py" recheck "$WORK/gate" --gate-log "$OUT/gate.jsonl" --json "$OUT/recheck.json" | tee "$OUT/recheck.txt"
src=${PIPESTATUS[0]}
[ -s "$OUT/time.jsonl" ] && python3 "$HERE/emu8.py" summarize "$OUT/time.jsonl" --json "$OUT/summary.json" | tee "$OUT/summary.md"
[ $rc -ne 0 ] && exit $rc
exit "$src"
