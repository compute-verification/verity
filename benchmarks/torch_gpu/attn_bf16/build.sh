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
#        the toolkit, default the first of /usr/local/cuda, /usr/local/cuda-13.0 with nvcc >= 12.8; CUBIN_ONLY=1 stops after the
#        SASS gates and needs no uv; FIXTURE_PY names a Python with numpy to run fixture.py with this tree's verity
#        instead of uv's, as on a GPU host's bench venv)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../ship.sh"
ship_cuda /usr/local/cuda /usr/local/cuda-13.0
OUT=${1:-/tmp/attn-bf16-ship}; shift || true
ship_fresh
ship_toolchain
ship_stage benchmarks/torch_gpu/attn_bf16
for ARCH in sm_90a sm_120a; do
  ship_cubin_flags "$ARCH"
  "$SHIP_NVCC" "${FLAGS[@]}" -o "$OUT/attn_bf16.$ARCH.cubin" "$SRC/benchmarks/torch_gpu/attn_bf16/attn_bf16.cu" \
    2> "$OUT/ptxas.$ARCH.txt" \
    || { cat "$OUT/ptxas.$ARCH.txt" >&2; exit 1; }
  "$CUDA/bin/cuobjdump" -sass "$OUT/attn_bf16.$ARCH.cubin" > "$OUT/attn_bf16.$ARCH.sass"
  python3 "$HERE/sass_gate.py" "$ARCH" "$OUT/ptxas.$ARCH.txt" "$OUT/attn_bf16.$ARCH.sass" "$OUT/sass.$ARCH.json"
  echo "ship_nvcc ${FLAGS[*]} -o attn_bf16.$ARCH.cubin attn_bf16.cu" >> "$OUT/toolchain.txt"
done
[ -n "${CUBIN_ONLY:-}" ] && exit 0
cd "$ROOT"
if [ -n "${FIXTURE_PY:-}" ]; then
  PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH} "$FIXTURE_PY" benchmarks/torch_gpu/attn_bf16/fixture.py "$OUT/check" "$@"
else
  uv run python benchmarks/torch_gpu/attn_bf16/fixture.py "$OUT/check" "$@"
fi
cp "$HERE/run.py" "$HERE/attn.py" "$HERE/gen.py" "$HERE/run.sh" "$OUT/"
ship_source benchmarks/torch_gpu/attn_bf16 verity
ship_manifest
