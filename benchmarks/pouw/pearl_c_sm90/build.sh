#!/usr/bin/env bash
# Pearl-C's sm_90a tree (the H100 and H200), built off the card:
#   pearl_c_sm90.cubin   pearl_c_sm90.cu with nvcc >= 12.8 for sm_90a (-fmad=false, no FTZ or fast-math flag, no -lineinfo,
#                        so the same bytes from any directory), refused unless ptxas reports no warning, spill or stack frame
#                        and the SASS has the intended tensor-core instructions and nothing lowered: every FP8 atom a
#                        QGMMA.64xNx32.F32.E4M3.E4M3 at its tile's N (4 per 128-deep stage), every peel step an
#                        HGMMA.64xNx16.F32.BF16 (4 in each gemm_pearlc and gemm_pearlc_s, none elsewhere), no other MMA form
#                        (sass_gate.py)
#   sass.json            registers and tensor-core, copy and local-memory instruction counts per kernel
#   harness_gate.json    benchmarks/pouw/harness/sass_gate.py's verdict on the cubin (.FTZ and FP32 MUFU outside its pinned
#                        IEEE sequences), recorded, not enforced here: its pins name sm_120a's ptxas arguments alone
#   check/               fixture.py's inputs and expected buffers (needs verity and numpy)
#   run.py, toolchain.txt (nvcc --version and the full compile command line), source.txt (the verity commit), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c_sm90/build.sh [OUT]   (default /tmp/pearl-c-sm90-ship; CUDA names the toolkit, default
#        /usr/local/cuda-13.0; CUBIN_ONLY=1 stops after the SASS gates, with the cubin, ptxas.txt and the gates' JSON, and
#        needs no uv)
# It builds in benchmarks/torch_gpu/ship.sh's frame, from a copy of its sources and with its ship_nvcc.sh: a rebuild at the
# same commit with the same toolkit gives the same cubin, and so does one after a source or a header only moved. The frame
# needs gcc and a kernel that lets a process turn address randomization off (setarch -R), and refuses to build without.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../../torch_gpu/ship.sh"
ship_cuda /usr/local/cuda-13.0
OUT=${1:-/tmp/pearl-c-sm90-ship}
ship_fresh
ship_cubin_flags sm_90a
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*) echo "an FTZ or fast-math flag: ${FLAGS[*]}" >&2; exit 1;; esac
ship_stage benchmarks/pouw/pearl_c_sm90 benchmarks/pouw/pearl_c_sm120 benchmarks/pouw/pearl_c verity/definitions/kernels
"$SHIP_NVCC" "${FLAGS[@]}" -o "$OUT/pearl_c_sm90.cubin" "$SRC/benchmarks/pouw/pearl_c_sm90/pearl_c_sm90.cu" 2> "$OUT/ptxas.txt" || { cat "$OUT/ptxas.txt" >&2; exit 1; }
PATH="$CUDA/bin:$PATH" "$CUDA/bin/cuobjdump" -sass "$OUT/pearl_c_sm90.cubin" > "$OUT/pearl_c_sm90.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/pearl_c_sm90.sass" "$OUT/sass.json"
rm "$OUT/pearl_c_sm90.sass"
POUW_CUDA_BIN="$CUDA/bin" POUW_SASS_GATE_CACHE=none python3 "$ROOT/benchmarks/pouw/harness/sass_gate.py" "$OUT/pearl_c_sm90.cubin" \
  --json "$OUT/harness_gate.json" && echo "harness gate: pass" || echo "harness gate: FAIL (recorded in harness_gate.json)"
[ -n "${CUBIN_ONLY:-}" ] && exit 0
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c_sm90/fixture.py "$OUT/check")
cp "$HERE/run.py" "$OUT/"
{ "$CUDA/bin/nvcc" --version; echo "ship_nvcc ${FLAGS[*]} -o pearl_c_sm90.cubin pearl_c_sm90.cu"; } > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c_sm90 benchmarks/pouw/pearl_c_sm120 benchmarks/pouw/pearl_c verity/protocols/pouw/verifier verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
