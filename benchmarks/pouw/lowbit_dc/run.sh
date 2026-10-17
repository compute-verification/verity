#!/usr/bin/env bash
# LOWBIT-DC's job on one GPU (B200 sm_100a or H200 sm_90a; the family is read from the device). Outside the lease: build
# fpx.cu and dump its SASS. Under one gpu-lease: `fpx run` (records.jsonl, windows.jsonl, summary.json). After it:
# analyze.py's tables (analysis.md, analysis.json).
#
#   research run --on vy-dc-b200 --project verity --campaign cmp-13 --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=lowbit-dc -- bash benchmarks/pouw/lowbit_dc/run.sh
# Exit: fpx's (0 every verdict as stated, 4 not), 1 build failed, 2 no device or no nvcc.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "run.sh: no nvcc" >&2; exit 2; }
export PATH="$(dirname "$NVCC"):$PATH"
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
case "${cap%%.*}" in
  9) ARCH=90a ;;
  10) ARCH=100a ;;
  *) echo "run.sh: fpx needs sm_90a or sm_100a, the device reports '$cap'" >&2; exit 2 ;;
esac
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" --version >"$OUT/nvcc.txt" 2>&1
"$NVCC" -O3 -std=c++17 -Xptxas -v -gencode "arch=compute_$ARCH,code=sm_$ARCH" "$HERE/fpx.cu" -o "$WORK/fpx" \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/fpx" >"$OUT/fpx.sass"
grep -oE "UTC[A-Z]*MMA[A-Z0-9_.]*|[A-Z]GMMA\.[0-9x]*\.[A-Z0-9_.]*" "$OUT/fpx.sass" | sort | uniq -c | tee "$OUT/mma-opcodes.txt"

lease=()
command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
"${lease[@]}" bash -c '
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "${GPU_LEASE_UUID:-0}" >"$2/gpu.csv" 2>&1
  "$1/fpx" run "$2"' _ "$WORK" "$OUT"
rc=$?
python3 "$HERE/analyze.py" "$OUT" --json "$OUT/analysis.json" >"$OUT/analysis.md" || rc=${rc:-1}
cat "$OUT/summary.json" 2>/dev/null
exit $rc
