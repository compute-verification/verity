#!/usr/bin/env bash
# Pearl-C's sm_120 tree, built off the card:
#   pearl_c_sm120.cubin  pearl_c_sm120.cu with nvcc >= 12.8 for sm_120a (-fmad=false, no -lineinfo, so the same bytes from any
#                        directory), refused unless ptxas reports no warning, spill or stack frame and the SASS has the
#                        intended tensor-core instructions and nothing lowered: every FP8 atom a QMMA.16832.F32.E4M3.E4M3
#                        (32 per 64-deep stage in the 128 x 128 GEMMs, 8 in the 64 x 64 and 128 x 32 ones, at least one in form_s5 and words), every
#                        peel step an HMMA.16816.F32.BF16 (64 in each gemm_pearlc, none elsewhere), no other MMA form
#   hash_sm120a.cubin    #449's ../pearl_c/hash.cu (hash.cuh's -h1 kernels: frame-b3 row and tile trees, keyed BLAKE3 digests
#                        and leaves), the same flags, refused on a ptxas warning, spill, stack frame or any MMA (sass_hash.json)
#   sass.json            registers and tensor-core, copy and local-memory instruction counts per kernel
#   check/               fixture.py's inputs and expected buffers (needs verity and numpy)
#   run.py, toolchain.txt, source.txt (the verity commit), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c_sm120/build.sh [OUT]   (default /tmp/pearl-c-sm120-ship; CUDA names the toolkit, default
#        /usr/local/cuda-12.9)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CUDA=${CUDA:-/usr/local/cuda-12.9}; OUT=${1:-/tmp/pearl-c-sm120-ship}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin \
  -o "$OUT/pearl_c_sm120.cubin" "$HERE/pearl_c_sm120.cu" 2> "$OUT/ptxas.txt"
PATH="$CUDA/bin:$PATH" "$CUDA/bin/cuobjdump" -sass "$OUT/pearl_c_sm120.cubin" > "$OUT/pearl_c_sm120.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/pearl_c_sm120.sass" "$OUT/sass.json"
rm "$OUT/pearl_c_sm120.sass"
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin \
  -o "$OUT/hash_sm120a.cubin" "$HERE/../pearl_c/hash.cu" 2> "$OUT/ptxas_hash.txt"
PATH="$CUDA/bin:$PATH" "$CUDA/bin/cuobjdump" -sass "$OUT/hash_sm120a.cubin" > "$OUT/hash_sm120a.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas_hash.txt" "$OUT/hash_sm120a.sass" "$OUT/sass_hash.json" --no-mma
rm "$OUT/hash_sm120a.sass"
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c_sm120/fixture.py "$OUT/check")
cp "$HERE/run.py" "$OUT/"
"$CUDA/bin/nvcc" --version | tail -2 > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c_sm120 benchmarks/pouw/pouw_hash benchmarks/pouw/pearl_c/hash.cu \
  benchmarks/pouw/pearl_c/hash.cuh benchmarks/pouw/pearl_c/hash_sm120.cuh protocols/pouw packages/verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
