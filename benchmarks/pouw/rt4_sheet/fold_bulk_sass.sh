#!/usr/bin/env bash
# fold_bulk.cu's kernels as cubins and SASS for the H200's, B200's and RTX PRO 6000's arch (sm_90a, sm_100a, sm_120a),
# for dc_price.py to price on DS-90 and DS-100. Build only, no device; on a node with CUDA 13.0:
#
#   research run --on vy-nebius-2 --project verity --campaign eps1 --source <tree> --cwd source --mem-gb 4 \
#     --timeout 900 -- bash benchmarks/pouw/rt4_sheet/fold_bulk_sass.sh
#
# On sm_90a the WIDE helpers are two 128-bit accesses (256-bit global memory is sm_100's and sm_120's).
# Exit: 0 built, 1 a build failed, 2 no nvcc.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT/sass" "$OUT/env"
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "fold_bulk_sass.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
git -C "$HERE" rev-parse HEAD >"$OUT/tree.sha" 2>/dev/null
rc=0
for arch in ${ARCHS:-sm_90a sm_100a sm_120a}; do
  "$NVCC" -O3 -std=c++17 -cubin -arch="$arch" -Xptxas -v "$HERE/fold_bulk.cu" -o "$OUT/fold_bulk_$arch.cubin" \
    >"$OUT/build_$arch.log" 2>&1 || { echo "$arch: build failed" >&2; tail -20 "$OUT/build_$arch.log" >&2; rc=1; continue; }
  "$(dirname "$NVCC")/cuobjdump" -sass "$OUT/fold_bulk_$arch.cubin" >"$OUT/sass/fold_bulk_$arch.sass" 2>&1
  echo "$arch: $(grep -c 'Function :' "$OUT/sass/fold_bulk_$arch.sass") functions, $(grep -c 'spill' "$OUT/build_$arch.log") spill lines, $(grep -E '[1-9][0-9]* bytes spill' -c "$OUT/build_$arch.log") nonzero"
done
exit $rc
