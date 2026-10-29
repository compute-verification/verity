#!/usr/bin/env bash
# Pearl-C4's shipped tree, built off the GPU box:
#   pearl_c4.cubin   pearl_c4.cu with CUDA 12.9's nvcc for sm_120a (-fmad=false, no -lineinfo), against GPU 2's hashing module
#                    pinned at HASH_COMMIT (benchmarks/pouw/pouw_hash/pouw_hash.cuh, its sha256 checked) and the tree's
#                    benchmarks/pouw/pearl_c/sha512.cuh (the seeds' and lines' SHA-512, sha512_rows' sha512x.cuh); refused unless
#                    sass.py passes (no warning, spill, stack or local memory; each kernel's OMMA, HMMA and E2M1 counts)
#   sass.json        registers and those counts per kernel
#   check/, check2/  fixture.py's buffers for NVFP4 and MXFP4 at 128x128x1024 and 256x384x2048 (needs verity and numpy)
#   dev.py, run.sh, toolchain.txt (nvcc's version and every flag, none of them FTZ or fast-math), source.txt (the verity
#                    commit and the header's), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c4/build.sh [OUT]   (default /tmp/pearl-c4-ship; C9 names CUDA 12.9, default /usr/local/cuda-12.9)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
C9=${C9:-/usr/local/cuda-12.9}; OUT=${1:-/tmp/pearl-c4-ship}
HASH_BRANCH=cursor/pouw-hash-sm120-9569
HASH_COMMIT=71086532
HASH_SHA256=9e861db9227b2267256cf642cbce06dfe807ddfa8b94db42f6dbb199ef1a63cc
[ -x "$C9/bin/nvcc" ] || { echo "no nvcc at $C9 (CUDA 12.9)" >&2; exit 1; }
"$C9/bin/nvcc" --version | grep -q "release 12.9" || { echo "$C9 is not CUDA 12.9" >&2; exit 1; }
git -C "$ROOT" cat-file -e "$HASH_COMMIT^{commit}" 2> /dev/null || git -C "$ROOT" fetch -q origin "$HASH_BRANCH"
rm -rf "$OUT"; mkdir -p "$OUT/inc"
git -C "$ROOT" show "$HASH_COMMIT:benchmarks/pouw/pouw_hash/pouw_hash.cuh" > "$OUT/inc/pouw_hash.cuh"
echo "$HASH_SHA256  $OUT/inc/pouw_hash.cuh" | sha256sum -c --quiet
FLAGS="-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin"
"$C9/bin/nvcc" $FLAGS -I "$OUT/inc" -o "$OUT/pearl_c4.cubin" "$HERE/pearl_c4.cu" 2> "$OUT/ptxas.txt"
"$C9/bin/cuobjdump" -sass "$OUT/pearl_c4.cubin" > "$OUT/pearl_c4.sass"
python3 "$HERE/sass.py" "$OUT/ptxas.txt" "$OUT/pearl_c4.sass" "$OUT/sass.json"
rm -r "$OUT/pearl_c4.sass" "$OUT/inc"
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c4/fixture.py "$OUT/check" &&
  uv run python benchmarks/pouw/pearl_c4/fixture.py "$OUT/check2" --shape 256x384x2048)
cp "$HERE/dev.py" "$HERE/run.sh" "$OUT/"
{ "$C9/bin/nvcc" --version | tail -2; echo "nvcc $FLAGS"; } > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c4 benchmarks/pouw/pearl_c/sha512.cuh benchmarks/torch_gpu/sha512_rows/sha512x.cuh verity/core/protocols/pouw verity
  echo "pouw_hash.cuh $HASH_COMMIT $HASH_SHA256"; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
