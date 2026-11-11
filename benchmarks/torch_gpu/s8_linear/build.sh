#!/usr/bin/env bash
# The exact linear's sm_90a tree (the H100 and H200), built off the card, and the ship run.sh unpacks there:
#   s8_wgmma.cubin   s8_wgmma.cu with nvcc >= 12.8 for sm_90a (-fmad=false, no FTZ or fast-math flag), refused unless
#                    sass_gate.py passes it: no ptxas warning, serialized wgmma, spill or local memory; every GEMM's MMAs its
#                    IGMMA.64xBNx32 s8 atom alone; the epilogue's s32-to-f32 conversion, FMUL and F2FP.BF16, no FFMA, no .FTZ
#   sass.json        the gate's record: registers, the atoms' SASS forms and instruction counts per kernel
#   row_table/       the commitment kernels for sm_90 (`row_tables_device.device_lib("90")`, commit.py's), in the layout of
#                    ~/.cache/verity/row_table, which run.sh seeds from it; skipped with no Python at $PY
#   toolchain.txt (nvcc --version and the command line), source.txt (the commit), MANIFEST.sha256
# Usage, from the tree's root on a CPU node with CUDA 13's nvcc:
#   uv sync --all-packages --extra torch-cpu && bash benchmarks/torch_gpu/s8_linear/build.sh /tmp/s8-linear-ship
# OUT (default /tmp/s8-linear-ship); CUDA names the toolkit (default /usr/local/cuda-13.0), PY the Python with verity_vllm
# (default the tree's .venv).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../ship.sh"
ship_cuda /usr/local/cuda-13.0
OUT=${1:-/tmp/s8-linear-ship}; PY=${PY:-$ROOT/.venv/bin/python}
ship_fresh
ship_toolchain
ship_cubin_flags sm_90a
ship_stage benchmarks/torch_gpu/s8_linear adapters/vllm/verity_vllm
"$SHIP_NVCC" "${FLAGS[@]}" -o "$OUT/s8_wgmma.cubin" "$SRC/benchmarks/torch_gpu/s8_linear/s8_wgmma.cu" 2> "$OUT/ptxas.txt" \
  || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$OUT/s8_wgmma.cubin" > "$OUT/s8_wgmma.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/s8_wgmma.sass" "$OUT/sass.json"
if [ -x "$PY" ]; then
  H=$(mktemp -d); trap 'rm -rf "$H"' EXIT
  HOME=$H PATH="$CUDA/bin:$PATH" PYTHONPATH="$SRC/adapters/vllm:$ROOT" "$PY" -c \
    "from verity_vllm.commit import row_tables_device as D; print(D.device_lib('90')._name)"
  cp -r "$H/.cache/verity/row_table" "$OUT/row_table"
else
  echo "no Python at $PY: no row_table build in the ship"
fi
echo "ship_nvcc ${FLAGS[*]} -o s8_wgmma.cubin s8_wgmma.cu" >> "$OUT/toolchain.txt"
ship_source benchmarks/torch_gpu/s8_linear adapters/vllm/verity_vllm/commit \
  adapters/vllm/verity_vllm/protocols/pouw/pouw_cuda
ship_manifest
