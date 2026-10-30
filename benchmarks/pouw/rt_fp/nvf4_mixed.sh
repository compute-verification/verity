#!/usr/bin/env bash
# The mxf4nvf4 exactness probe on one B200 (sm_100a) or B300 (sm_103a): build nvf4_mixed.cu, dump its SASS and MMA
# opcodes, then under one gpu-lease `nvf4_mixed run` (calibration.json, records.jsonl, summary.json).
#
#   research run --on vy-dc-b200 --project verity --campaign eps1 --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/rt_fp/nvf4_mixed.sh
# Exit: nvf4_mixed's (0 every verdict as stated, 3 no block-to-byte scale map, 4 not as stated), 1 build failed, 2 no
# device or no nvcc.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "nvf4_mixed.sh: no nvcc" >&2; exit 2; }
export PATH="$(dirname "$NVCC"):$PATH"
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
case "$cap" in
  10.0) ARCH=100a ;;
  10.3) ARCH=103a ;;
  *) echo "nvf4_mixed.sh: needs sm_100a or sm_103a, the device reports '$cap'" >&2; exit 2 ;;
esac
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" --version >"$OUT/nvcc.txt" 2>&1
"$NVCC" -O3 -std=c++17 -Xptxas -v -gencode "arch=compute_$ARCH,code=sm_$ARCH" "$HERE/nvf4_mixed.cu" -o "$WORK/nvf4_mixed" \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
cuobjdump -sass "$WORK/nvf4_mixed" >"$OUT/nvf4_mixed.sass"
grep -oE "UTC[A-Z]*MMA[A-Z0-9_.]*" "$OUT/nvf4_mixed.sass" | sort | uniq -c | tee "$OUT/mma-opcodes.txt"
"$WORK/nvf4_mixed" list
lease=()
command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
"${lease[@]}" bash -c 'nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit \
  --format=csv -i "${GPU_LEASE_UUID:-0}" >"$2/gpu.csv" 2>&1; "$1/nvf4_mixed" run "$2"' _ "$WORK" "$OUT"
rc=$?
cat "$OUT/calibration.json" "$OUT/summary.json" 2>/dev/null
exit $rc
