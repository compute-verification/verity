#!/usr/bin/env bash
# sha.cu on the H200: build and dump the SASS outside the lease, then run under one gpu-lease (sha.cu: SHA-512 compute-bound).
#   research run --on vy-dc-h200-5 --project verity --campaign pouw-night --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=n-floor -- bash benchmarks/pouw/np_floor/sha.sh
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" -O3 -arch=sm_90a -std=c++17 "$HERE/sha.cu" -o "$WORK/sha"
"$(dirname "$NVCC")/cuobjdump" -sass "$WORK/sha" >"$OUT/sha.sass"
gpu-lease 1 --wait --max-min 5 -- bash -c '
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw --format=csv -lms 200 >"'"$OUT"'/smi.csv" &
  p=$!
  "'"$WORK"'/sha" | tee "'"$OUT"'/sha.json"
  kill $p'
