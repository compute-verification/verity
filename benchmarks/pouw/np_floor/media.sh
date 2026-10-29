#!/usr/bin/env bash
# nvjpg.cu on the H200: the JPEG decode engines' output rate through nvJPEG's hardware backend, with the engines'
# utilization sampled (nvidia-smi dmon's jpg column) to show the decode ran on them. nvJPEG from the node's toolkit, else
# from the CUDA wheels in a scratch venv.
#   research run --on vy-dc-h200-5 --project verity --campaign pouw-night --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=n-floor -- bash benchmarks/pouw/np_floor/media.sh [W] [B] [ITERS]
# Writes $OUT/build.log, nvjpg.json, dmon.txt and smi.csv.
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "media.sh: no nvcc" >&2; exit 2; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
CU=$(cd "$(dirname "$NVCC")/.." && pwd)
INC=(-I"$CU/include") LIB=(-L"$CU/lib64" -Xlinker -rpath -Xlinker "$CU/lib64")
if [ ! -f "$CU/include/nvjpeg.h" ]; then
  python3 -m venv "$WORK/v"
  "$WORK/v/bin/pip" install -q nvidia-nvjpeg-cu12 >>"$OUT/build.log" 2>&1 || true
  h=$(find "$WORK/v" -name nvjpeg.h | head -1)
  [ -n "$h" ] || { echo "media.sh: no nvjpeg.h in $CU or the wheels" | tee -a "$OUT/build.log" >&2; exit 2; }
  l=$(dirname "$(find "$WORK/v" -name 'libnvjpeg.so*' | head -1)")
  INC+=(-I"$(dirname "$h")") LIB+=(-L"$l" -Xlinker -rpath -Xlinker "$l")
fi
build() { "$NVCC" -O2 -std=c++17 -arch=sm_90a "$@" "${INC[@]}" "$HERE/nvjpg.cu" -o "$WORK/nvjpg" "${LIB[@]}" -lnvjpeg; }
build -DHWINFO >>"$OUT/build.log" 2>&1 || build >>"$OUT/build.log" 2>&1
gpu-lease 1 --wait --max-min 6 -- bash -c '
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw --format=csv -lms 200 >"$0/smi.csv" &
  p=$!
  nvidia-smi dmon -i "$GPU_LEASE_UUID" -s u -o T >"$0/dmon.txt" 2>&1 &
  q=$!
  "$1" "${@:2}" | tee "$0/nvjpg.json"
  kill $p $q' "$OUT" "$WORK/nvjpg" "$@"
