#!/usr/bin/env bash
# FOLD-XOR's device check (fold_xor.cu): the 78-instruction IMAD-pipe XOR built for sm_100a, its SASS dumped, and
# bit-exact on the device. From a pushed commit:
#
#   research run --on vy-dc-b200 --project verity --campaign eps1 --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=<id> -- bash benchmarks/pouw/theory/fold_xor.sh
#
# The SASS ($OUT/sass/fold_xor.sass) is counted by census class off the node: `python fold_xor.py --sass FILE`.
# Exit: 0 done, 1 build failed, 2 no nvcc, 3 not bit-exact.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "${FOLD_LEASED:-}" != 1 ]; then
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "fold_xor.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  BIN="$OUT/bin/fold_xor"
  "$NVCC" -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v "$HERE/fold_xor.cu" -o "$BIN" \
    >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  sha256sum "$BIN" >"$OUT/bin/fold_xor.sha256"
  "$(dirname "$NVCC")/cuobjdump" -sass "$BIN" >"$OUT/sass/fold_xor.sass" 2>&1
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 20 --)
  [ ${#lease[@]} -eq 0 ] && echo "fold_xor.sh: no gpu-lease on this node; running on its GPU directly" >&2
  "${lease[@]}" env FOLD_LEASED=1 BIN="$BIN" bash "${BASH_SOURCE[0]}"
  exit $?
fi

# ---- leased ------------------------------------------------------------------------------------------------------------
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
"$BIN" | tee "$OUT/device.txt"
exit "${PIPESTATUS[0]}"
