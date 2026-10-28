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
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CUDA=${CUDA:-/usr/local/cuda-13.0}; OUT=${1:-/tmp/s8-linear-ship}; PY=${PY:-$ROOT/.venv/bin/python}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA: set CUDA to a CUDA 13 toolkit" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -Xptxas -v -cubin)
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*) echo "an FTZ or fast-math flag: ${FLAGS[*]}" >&2; exit 1;; esac
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/s8_wgmma.cubin" "$HERE/s8_wgmma.cu" 2> "$OUT/ptxas.txt" || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$OUT/s8_wgmma.cubin" > "$OUT/s8_wgmma.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/s8_wgmma.sass" "$OUT/sass.json"
if [ -x "$PY" ]; then
  H=$(mktemp -d); trap 'rm -rf "$H"' EXIT
  HOME=$H PATH="$CUDA/bin:$PATH" PYTHONPATH="$ROOT/integrations/vllm:$ROOT" "$PY" -c \
    "from verity_vllm.commit import row_tables_device as D; print(D.device_lib('90')._name)"
  cp -r "$H/.cache/verity/row_table" "$OUT/row_table"
else
  echo "no Python at $PY: no row_table build in the ship"
fi
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o s8_wgmma.cubin s8_wgmma.cu"; } > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"
  git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/s8_linear integrations/vllm/verity_vllm/commit \
    integrations/vllm/verity_vllm/protocol_options/pouw_cuda 2>/dev/null || true; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "ship:"
