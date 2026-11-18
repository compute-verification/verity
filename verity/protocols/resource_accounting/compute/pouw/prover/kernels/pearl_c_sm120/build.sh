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
#   ptxas.txt, toolchain.txt (nvcc --version and the full nvcc command line)
# ship.sh adds the check and what the card runs it with.  Needs no uv.
# Usage: bash verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120/build.sh [OUT]   (default /tmp/pearl-c-sm120-ship; CUDA_HOME names the toolkit and
#        GENCODE the target, which callers read from the machine and its card; a card of another target is refused)
#        bash .../build.sh --arch   prints the target the tree's kernels are built for and builds nothing (needs no toolkit)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=sm_120a   # pearl_c_sm120.cu's kernels are this target's alone (sass_gate.py holds its MMA forms)
[ "${1:-}" = --arch ] && { echo "$ARCH"; exit 0; }
CUDA=${CUDA_HOME:?set CUDA_HOME to a CUDA toolkit of nvcc 12.8 or later}; OUT=${1:-/tmp/pearl-c-sm120-ship}
GENCODE=${GENCODE:?set GENCODE to the -gencode value of the target (benchmarks/kernels/machine.py gencode ARCH)}
case $GENCODE in *,code=$ARCH) ;; *) echo "pearl_c_sm120 is $ARCH's: GENCODE=$GENCODE is another card's target" >&2; exit 2 ;; esac
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -gencode "$GENCODE" -fmad=false -Xptxas -v -cubin)
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*) echo "an FTZ or fast-math flag: ${FLAGS[*]}" >&2; exit 1;; esac
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/pearl_c_sm120.cubin" "$HERE/pearl_c_sm120.cu" 2> "$OUT/ptxas.txt"
PATH="$CUDA/bin:$PATH" "$CUDA/bin/cuobjdump" -sass "$OUT/pearl_c_sm120.cubin" > "$OUT/pearl_c_sm120.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/pearl_c_sm120.sass" "$OUT/sass.json"
rm "$OUT/pearl_c_sm120.sass"
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o pearl_c_sm120.cubin pearl_c_sm120.cu"; } > "$OUT/toolchain.txt"
