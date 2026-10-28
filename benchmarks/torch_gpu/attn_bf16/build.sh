#!/usr/bin/env bash
# attn_bf16's ship, built off the card on a CPU node with the CUDA toolkit:
#   attn_bf16.{sm_90a,sm_120a}.cubin   attn_bf16.cu with nvcc >= 12.8 (-fmad=false, no FTZ or fast-math flag, no -lineinfo,
#                     so the same bytes from any directory), each refused unless ptxas reports no warning, spill or stack frame
#                     and the SASS's only floating-point instructions are HMMA.16816.F32.BF16 and the cvt (sass_gate.py)
#   sass.<arch>.json, ptxas.<arch>.txt   registers and HMMA, F2FP and local-memory counts per kernel; ptxas's report
#   check/            fixture.py's cases: inputs and the buffers the kernels must leave, and FULL's digests (needs verity)
#   run.py, attn.py, gen.py, run.sh, toolchain.txt (nvcc --version and the nvcc command lines), source.txt (the verity commit),
#   MANIFEST.sha256
# Usage: bash benchmarks/torch_gpu/attn_bf16/build.sh [OUT [CASE ...]]   (default /tmp/attn-bf16-ship, every case; CUDA names
#        the toolkit, default the first of /usr/local/cuda, /usr/local/cuda-13.0 with nvcc; CUBIN_ONLY=1 stops after the
#        SASS gates and needs no uv; FIXTURE_PY names a Python with numpy to run fixture.py with this tree's verity
#        instead of uv's, as on a GPU host's bench venv)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
if [ -z "${CUDA:-}" ]; then
  for c in /usr/local/cuda /usr/local/cuda-13.0; do [ -x "$c/bin/nvcc" ] && CUDA=$c && break; done
fi
CUDA=${CUDA:-/usr/local/cuda}; OUT=${1:-/tmp/attn-bf16-ship}; shift || true
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
"$CUDA/bin/nvcc" --version > "$OUT/toolchain.txt"
for ARCH in sm_90a sm_120a; do
  FLAGS=(-O3 -std=c++17 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" -fmad=false -Xptxas -v -cubin)
  case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*|*prec-div=false*|*lineinfo*) echo "a refused flag: ${FLAGS[*]}" >&2; exit 1;; esac
  "$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/attn_bf16.$ARCH.cubin" "$HERE/attn_bf16.cu" 2> "$OUT/ptxas.$ARCH.txt" \
    || { cat "$OUT/ptxas.$ARCH.txt" >&2; exit 1; }
  "$CUDA/bin/cuobjdump" -sass "$OUT/attn_bf16.$ARCH.cubin" > "$OUT/attn_bf16.$ARCH.sass"
  python3 "$HERE/sass_gate.py" "$ARCH" "$OUT/ptxas.$ARCH.txt" "$OUT/attn_bf16.$ARCH.sass" "$OUT/sass.$ARCH.json"
  echo "nvcc ${FLAGS[*]} -o attn_bf16.$ARCH.cubin attn_bf16.cu" >> "$OUT/toolchain.txt"
done
[ -n "${CUBIN_ONLY:-}" ] && exit 0
cd "$ROOT"
if [ -n "${FIXTURE_PY:-}" ]; then
  PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH} "$FIXTURE_PY" benchmarks/torch_gpu/attn_bf16/fixture.py "$OUT/check" "$@"
else
  uv run python benchmarks/torch_gpu/attn_bf16/fixture.py "$OUT/check" "$@"
fi
cp "$HERE/run.py" "$HERE/attn.py" "$HERE/gen.py" "$HERE/run.sh" "$OUT/"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/attn_bf16 verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
