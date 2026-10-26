#!/usr/bin/env bash
# Pearl-C's sm_120 tree, built off the card:
#   pearl_c_sm120.cubin  pearl_c_sm120.cu with nvcc >= 12.8 for sm_120a (-fmad=false, no FTZ or fast-math flag, no -lineinfo,
#                        so the same bytes from any directory), refused unless ptxas reports no warning, spill or stack frame
#                        and the SASS has the intended tensor-core instructions and nothing lowered: every FP8 atom a
#                        QMMA.16832.F32.E4M3.E4M3 (64 per 128-deep stage in the 128 x 128 GEMMs on the mainloop, 4 per k32
#                        in the 64 x 64 and 128 x 32 ones, at least one in form_s5 and words), every peel step an
#                        HMMA.16816.F32.BF16 (64 in each 128 x 128 gemm_pearlc and gemm_pearlc_s, 16 in the 64 x 64 ones,
#                        none elsewhere), no other MMA
#                        form
#   sass.json            registers and tensor-core, copy and local-memory instruction counts per kernel
#   check/               fixture.py's inputs and expected buffers (needs verity and numpy)
#   run.py, toolchain.txt (nvcc --version and the full nvcc command line), source.txt (the verity commit), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c_sm120/build.sh [OUT]   (default /tmp/pearl-c-sm120-ship; CUDA names the toolkit, default
#        /usr/local/cuda-13.0; CUBIN_ONLY=1 stops after the SASS gate, with the cubin, ptxas.txt and sass.json, and needs no uv)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CUDA=${CUDA:-/usr/local/cuda-13.0}; OUT=${1:-/tmp/pearl-c-sm120-ship}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin)
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*) echo "an FTZ or fast-math flag: ${FLAGS[*]}" >&2; exit 1;; esac
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/pearl_c_sm120.cubin" "$HERE/pearl_c_sm120.cu" 2> "$OUT/ptxas.txt"
PATH="$CUDA/bin:$PATH" "$CUDA/bin/cuobjdump" -sass "$OUT/pearl_c_sm120.cubin" > "$OUT/pearl_c_sm120.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/pearl_c_sm120.sass" "$OUT/sass.json"
rm "$OUT/pearl_c_sm120.sass"
[ -n "${CUBIN_ONLY:-}" ] && exit 0
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c_sm120/fixture.py "$OUT/check")
cp "$HERE/run.py" "$OUT/"
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o pearl_c_sm120.cubin pearl_c_sm120.cu"; } > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c_sm120 benchmarks/pouw/pearl_c verity/core/protocols/pouw verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
