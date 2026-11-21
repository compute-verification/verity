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
#   sources.txt          the repository's files the cubin is made from (../pearl_c/build.sh's pearl_c_sources)
# ship.sh adds the check and what the card runs it with.  Needs no uv.
# Usage: bash verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm120/build.sh [OUT]   (default /tmp/pearl-c-sm120-ship; CUDA_HOME names the toolkit and
#        GENCODE the target, which callers read from the machine and its card; a card of another target is refused)
#        bash .../build.sh --arch   prints the target the tree's kernels are built for and builds nothing (needs no toolkit)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=sm_120a   # pearl_c_sm120.cu's kernels are this target's alone (sass_gate.json holds its MMA forms)
[ "${1:-}" = --arch ] && { echo "$ARCH"; exit 0; }
. "$HERE/../pearl_c/build.sh"; pearl_c_build "$HERE" "$ARCH" "${1:-}"
