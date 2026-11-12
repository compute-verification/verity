#!/usr/bin/env bash
# The units' ship, built off the card on a CPU node with the CUDA toolkit, for each arch A of sm_90a (the H200) and sm_120a
# (the RTX PRO 6000), with one command line but for the arch:
#   units.A.cubin         units.cu with nvcc >= 12.8 (-fmad=false, no FTZ or fast-math flag, no -lineinfo; 13.0 first, whose
#                         cubins carry the tool note the harness SASS gate reads), refused unless build_check.py passes it
#   build-check.A.json, ptxas.A.txt, units.A.sass   the check's record, ptxas's report, the SASS
#   harness-gate.A.json, harness-gate.A.txt   the harness SASS gate's verdict (benchmarks/pouw/harness/sass_gate.py),
#                         recorded and never refusing: 13.0.1-sm_90a excepts the H200's MUFU.EX2, and no toolkit excepts
#                         sm_120a's MUFU.EX2 or the sweep's bare MUFU.RSQ (ops 9 and 10) yet
#   check/                fixture.py's pools, words and sweep digests from the twins (needs verity: run under uv); REPLAY
#                         names the CPU replay for fixture.py's cross-check (default the tree's archive/examples/h200/replay.py),
#                         ROWS an ops/rows.py to load in place of the installed one
#   the unit modules, lib.py, bench.py, check.py, settings.py (their registry, which they find beside them),
#   toolchain.txt (nvcc --version and the command lines), source.txt (the commit), MANIFEST.sha256
# Usage: bash benchmarks/torch_gpu/units/build.sh [OUT]   (default /tmp/units-ship; CUDA names the toolkit, default the first
#        of /usr/local/cuda-13.0, /usr/local/cuda with nvcc >= 12.8; CUBIN_ONLY=1 stops after the checks and needs no uv; PROCS the
#        fixture's processes)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../ship.sh"
ship_cuda /usr/local/cuda-13.0 /usr/local/cuda
OUT=${1:-/tmp/units-ship}
ship_fresh
ship_toolchain
ship_stage benchmarks/torch_gpu/units
for A in 90a 120a; do
  ship_cubin_flags "sm_$A"
  "$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/units.sm_$A.cubin" "$SRC/benchmarks/torch_gpu/units/units.cu" \
    2> "$OUT/ptxas.sm_$A.txt" \
    || { cat "$OUT/ptxas.sm_$A.txt" >&2; exit 1; }
  "$CUDA/bin/cuobjdump" -sass "$OUT/units.sm_$A.cubin" > "$OUT/units.sm_$A.sass"
  python3 "$HERE/build_check.py" "$OUT/ptxas.sm_$A.txt" "$OUT/units.sm_$A.sass" "$OUT/build-check.sm_$A.json" "sm_$A"
  set +e
  POUW_CUDA_BIN="$CUDA/bin" POUW_SASS_GATE_CACHE=none python3 "$ROOT/benchmarks/pouw/harness/sass_gate.py" \
    "$OUT/units.sm_$A.cubin" --json "$OUT/harness-gate.sm_$A.json" > "$OUT/harness-gate.sm_$A.txt" 2>&1
  echo "harness sass_gate.py rc $?" >> "$OUT/harness-gate.sm_$A.txt"
  set -e
  tail -3 "$OUT/harness-gate.sm_$A.txt"
  echo "nvcc ${FLAGS[*]} -o units.sm_$A.cubin units.cu" >> "$OUT/toolchain.txt"
done
cp "$HERE"/{words,lib,rmsnorm,quant,softmax,argmax,relu,add,sweep,bench,check,settings}.py "$OUT/"
ship_source benchmarks/torch_gpu/units verity
REPLAY=${REPLAY:-$ROOT/archive/examples/h200/replay.py}
if [ -z "${CUBIN_ONLY:-}" ]; then
  (cd "$ROOT" && uv run python "$HERE/fixture.py" "$OUT/check" --replay "$REPLAY" ${ROWS:+--rows "$ROWS"})
fi
ship_manifest
