#!/usr/bin/env bash
# libvc.so into DIR (default build/ beside this file) for ARCHS (default this machine's GPU; "90 100 120" for all three), each
# as sm_XXa code. VC_NVCC_FLAGS adds flags (e.g. -DVC_LEAF_MIN_BLOCKS=1).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${1:-$HERE/build}
CUDA=${CUDA:-/usr/local/cuda}
ARCHS=${ARCHS:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)}
mkdir -p "$OUT_DIR"
GEN=()
for a in $ARCHS; do GEN+=(-gencode "arch=compute_${a}a,code=sm_${a}a"); done
# shellcheck disable=SC2086
"$CUDA/bin/nvcc" -O3 -std=c++17 --threads 0 -shared -Xcompiler -fPIC -cudart shared "${GEN[@]}" -Xptxas -v ${VC_NVCC_FLAGS:-} \
  "$HERE/vc.cu" -o "$OUT_DIR/libvc.so"
