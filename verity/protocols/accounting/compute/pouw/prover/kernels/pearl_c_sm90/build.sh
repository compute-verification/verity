#!/usr/bin/env bash
# Pearl-C's sm_90a tree (the H100 and H200), built off the card:
#   pearl_c_sm90.cubin   pearl_c_sm90.cu for sm_90a with an nvcc whose ptxas sass_pins.json pins for sm_90a (any other is
#                        refused by name) (-fmad=false, no FTZ or fast-math flag, no -lineinfo,
#                        so the same bytes from any directory), refused unless ptxas reports no warning, spill or stack frame
#                        and the SASS has the intended tensor-core instructions and nothing lowered: every FP8 atom a
#                        QGMMA.64xNx32.F32.E4M3.E4M3 at its tile's N (4 per 128-deep stage), every peel step an
#                        HGMMA.64xNx16.F32.BF16 (4 in each gemm_pearlc and gemm_pearlc_s, none elsewhere), no other MMA form
#                        (sass_gate.json, through ../../build_gate.py)
#   sass.json            registers and tensor-core, copy and local-memory instruction counts per kernel
#   harness_gate.json    verity/protocols/accounting/compute/pouw/prover/sass_gate.py's verdict on the cubin (.FTZ and FP32 MUFU outside its pinned
#                        IEEE sequences, under its sm_90a pins), refused unless it passes
#   debit/               debit_sm90.cubin: debit_sm90.cu (the words' debit counts, debit.py's) with the same flags, refused
#                        unless ptxas reports no spill or stack frame and its every MMA is a QGMMA.64x32x32.F32.E4M3.E4M3
#                        (debit_ptxas.txt, debit_mma.txt), in a directory of its own: one Kernel.load reads holds one cubin
#   ptxas.txt, toolchain.txt (nvcc --version and the full nvcc command line)
#   sources.txt          the repository's files the cubins are made from (../pearl_c/build.sh's pearl_c_sources)
# ship.sh adds the check and what the card runs it with.  Needs no uv.
# Usage: bash verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm90/build.sh [OUT]   (default /tmp/pearl-c-sm90-ship; CUDA_HOME names the toolkit and
#        GENCODE the target, which callers read from the machine and its card; a card of another target is refused)
#        bash .../build.sh --arch   prints the target the tree's kernels are built for and builds nothing (needs no toolkit)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARCH=sm_90a   # pearl_c_sm90.cu's and debit_sm90.cu's kernels are this target's alone (sass_gate.json holds its MMA forms)
[ "${1:-}" = --arch ] && { echo "$ARCH"; exit 0; }
. "$HERE/../pearl_c/build.sh"
PINS=$ROOT/verity/protocols/accounting/compute/pouw/prover/sass_pins.json
PINNED=$(python3 -c 'import json, sys
print("\n".join(sorted({t["version"] for t in json.load(open(sys.argv[1]))["toolkits"].values()
                        if any(a.split()[:2] == ["-arch", sys.argv[2]] for a in t["args"])})))' "$PINS" "$ARCH")
NVCC=${CUDA_HOME:?set CUDA_HOME to the toolkit whose nvcc --version says: $PINNED}/bin/nvcc
[ -x "$NVCC" ] || { echo "no nvcc at $CUDA_HOME" >&2; exit 1; }
SAID=$("$NVCC" --version | grep '^Cuda compilation tools' || true)
grep -Fxq -- "$SAID" <<< "$PINNED" || { echo "pearl_c_sm90 builds only with the nvcc whose ptxas sass_pins.json pins for $ARCH," \
  "'$PINNED'; $NVCC says '${SAID:-no version}'. Set CUDA_HOME to that toolkit's directory" >&2; exit 1; }
pearl_c_build "$HERE" "$ARCH" "${1:-}"
POUW_SASS_GATE_CACHE=none python3 "$ROOT/verity/protocols/accounting/compute/pouw/prover/sass_gate.py" --cuda-bin "$CUDA/bin" "$OUT/pearl_c_sm90.cubin" \
  --json "$OUT/harness_gate.json" || { echo "pearl_c_sm90: the harness's SASS gate refused the cubin (harness_gate.json)" >&2; exit 1; }
echo "harness gate: pass"
B=$OUT/debit; mkdir -p "$B"
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$B/debit_sm90.cubin" "$HERE/debit_sm90.cu" 2> "$B/debit_ptxas.txt" || { cat "$B/debit_ptxas.txt" >&2; exit 1; }
awk '/stack frame/ && !/ 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads/ { bad = 1 } END { exit !bad }' \
  "$B/debit_ptxas.txt" && { echo "debit_sm90: a stack frame or a spill" >&2; cat "$B/debit_ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$B/debit_sm90.cubin" | grep -oE "[A-Z]*MMA[.A-Z0-9x]*" | sort | uniq -c > "$B/debit_mma.txt"
awk '$2 != "QGMMA.64x32x32.F32.E4M3.E4M3" { bad = 1 } { n += $1 } END { exit bad || n < 36 }' "$B/debit_mma.txt" \
  || { echo "debit_sm90: an MMA other than QGMMA.64x32x32.F32.E4M3.E4M3, or fewer than its 36 atoms:" >&2; cat "$B/debit_mma.txt" >&2; exit 1; }
pearl_c_sources "$HERE/debit_sm90.cu"
