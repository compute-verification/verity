#!/usr/bin/env bash
# engines.cu on the H200: build and dump its SASS outside the lease, check that every timed loop holds the instruction
# it prices (UBLK* / LDSM / LDS.128), then run under one gpu-lease with the SM clock sampled.
#   research run --on vy-dc-h200-5 --project verity --campaign pouw-night --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=n-floor -- bash benchmarks/pouw/np_floor/engines.sh
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" -O3 -arch=sm_90a -std=c++17 "$HERE/engines.cu" -o "$WORK/engines"
"$(dirname "$NVCC")/cuobjdump" -sass "$WORK/engines" >"$OUT/engines.sass"
for pat in UBLK LDSM LDS.128; do
  n=$(grep -c "$pat" "$OUT/engines.sass" || true)
  echo "sass $pat: $n" | tee -a "$OUT/gate.txt"
  [ "$n" -gt 0 ] || { echo "engines.sh: no $pat in the SASS" >&2; exit 1; }
done
gpu-lease 1 --wait --max-min 6 -- bash -c '
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw --format=csv -lms 200 >"'"$OUT"'/smi.csv" &
  p=$!
  "'"$WORK"'/engines" | tee "'"$OUT"'/engines.json"
  kill $p'
