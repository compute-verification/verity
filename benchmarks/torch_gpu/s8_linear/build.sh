#!/usr/bin/env bash
# The exact linear's sm_90a tree (the H100 and H200), built off the card, and the ship run.sh unpacks there:
#   s8_wgmma.cubin   s8_wgmma.cu with nvcc >= 12.8 for sm_90a (-fmad=false, no FTZ or fast-math flag), refused unless
#                    sass_gate.py passes it: no ptxas warning, serialized wgmma, spill or local memory; every GEMM's MMAs its
#                    IGMMA.64xBNx32 s8 atom alone; the epilogue's s32-to-f32 conversion, FMUL and F2FP.BF16, no FFMA, no .FTZ
#   sass.json        the gate's record: registers, the atoms' SASS forms and instruction counts per kernel
#   toolchain.txt (nvcc --version and the command line), source.txt (the commit), MANIFEST.sha256
# Usage, from the tree's root on a CPU node with CUDA 13's nvcc:
#   bash benchmarks/torch_gpu/s8_linear/build.sh /tmp/s8-linear-ship
# OUT (default /tmp/s8-linear-ship); CUDA names the toolkit (default /usr/local/cuda-13.0).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../ship.sh"
ship_cuda /usr/local/cuda-13.0
OUT=${1:-/tmp/s8-linear-ship}
ship_fresh
ship_toolchain
ship_cubin_flags sm_90a
ship_stage benchmarks/torch_gpu/s8_linear
"$SHIP_NVCC" "${FLAGS[@]}" -o "$OUT/s8_wgmma.cubin" "$SRC/benchmarks/torch_gpu/s8_linear/s8_wgmma.cu" 2> "$OUT/ptxas.txt" \
  || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$OUT/s8_wgmma.cubin" > "$OUT/s8_wgmma.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/s8_wgmma.sass" "$OUT/sass.json"
echo "ship_nvcc ${FLAGS[*]} -o s8_wgmma.cubin s8_wgmma.cu" >> "$OUT/toolchain.txt"
ship_source benchmarks/torch_gpu/s8_linear
ship_manifest
