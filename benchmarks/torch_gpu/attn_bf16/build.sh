#!/usr/bin/env bash
# attn_bf16's sm_120 tree, built off the card:
#   attn_bf16.cubin   attn_bf16.cu with nvcc >= 12.8 for sm_120a (-fmad=false, no FTZ or fast-math flag, no -lineinfo, so
#                     the same bytes from any directory), refused unless ptxas reports no warning, spill or stack frame and
#                     the SASS's only floating-point instructions are HMMA.16816.F32.BF16 and the cvt (sass_gate.py)
#   sass.json         registers and HMMA, F2FP and local-memory counts per kernel
#   check/            fixture.py's cases: inputs and the buffers the kernels must leave (needs verity and numpy)
#   run.py, toolchain.txt (nvcc --version and the full nvcc command line), source.txt (the verity commit), MANIFEST.sha256
# Usage: bash benchmarks/torch_gpu/attn_bf16/build.sh [OUT [CASE ...]]   (default /tmp/attn-bf16-ship, every case; CUDA names
#        the toolkit, default /usr/local/cuda; CUBIN_ONLY=1 stops after the SASS gate and needs no uv)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CUDA=${CUDA:-/usr/local/cuda}; OUT=${1:-/tmp/attn-bf16-ship}; shift || true
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin)
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*|*prec-div=false*|*lineinfo*) echo "a refused flag: ${FLAGS[*]}" >&2; exit 1;; esac
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/attn_bf16.cubin" "$HERE/attn_bf16.cu" 2> "$OUT/ptxas.txt"
"$CUDA/bin/cuobjdump" -sass "$OUT/attn_bf16.cubin" > "$OUT/attn_bf16.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/attn_bf16.sass" "$OUT/sass.json"
[ -n "${CUBIN_ONLY:-}" ] && exit 0
(cd "$ROOT" && uv run python benchmarks/torch_gpu/attn_bf16/fixture.py "$OUT/check" "$@")
cp "$HERE/run.py" "$OUT/"
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o attn_bf16.cubin attn_bf16.cu"; } > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/attn_bf16 verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 ! -path './check/*' | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
