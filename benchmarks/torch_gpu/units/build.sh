#!/usr/bin/env bash
# The units' ship, built off the card on a CPU node with the CUDA toolkit:
#   units.sm_90a.cubin    units.cu with nvcc >= 12.8 (-fmad=false, no FTZ or fast-math flag, no -lineinfo; 13.0 first, whose
#                         sm_90a cubins carry the tool note the harness SASS gate reads), refused unless build_check.py
#                         passes it
#   build-check.json, ptxas.txt, units.sm_90a.sass   the check's record, ptxas's report, the SASS
#   harness-gate.json, harness-gate.txt   the harness SASS gate's verdict (benchmarks/pouw/harness/sass_gate.py), recorded and
#                         never refusing: it has no sm_90a pin yet, and MUFU.EX2 has no exemption
#   check/                fixture.py's pools, words and sweep digests from the twins (needs verity: run under uv); REPLAY
#                         names the CMP-96 replay (e2e_h200's replay.py, #4460) for fixture.py's cross-check, default a
#                         replay.py the run was sent; ROWS names ops/rows.py (CIR-104) while main hasn't it, default a
#                         rows.py the run was sent
#   the unit modules, lib.py, bench.py, check.py, toolchain.txt (nvcc --version and the command line), source.txt (the commit),
#   MANIFEST.sha256
# Usage: bash benchmarks/torch_gpu/units/build.sh [OUT]   (default /tmp/units-ship; CUDA names the toolkit, default the first
#        of /usr/local/cuda-13.0, /usr/local/cuda with nvcc; CUBIN_ONLY=1 stops after the checks and needs no uv; PROCS the
#        fixture's processes)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
if [ -z "${CUDA:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && CUDA=$c && break; done
fi
CUDA=${CUDA:-/usr/local/cuda}; OUT=${1:-/tmp/units-ship}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -Xptxas -v -cubin)
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*|*lineinfo*|*prec-div*|*prec-sqrt*) echo "a refused flag: ${FLAGS[*]}" >&2; exit 1;; esac
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/units.sm_90a.cubin" "$HERE/units.cu" 2> "$OUT/ptxas.txt" \
  || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$OUT/units.sm_90a.cubin" > "$OUT/units.sm_90a.sass"
python3 "$HERE/build_check.py" "$OUT/ptxas.txt" "$OUT/units.sm_90a.sass" "$OUT/build-check.json"
set +e
POUW_CUDA_BIN="$CUDA/bin" POUW_SASS_GATE_CACHE=none python3 "$ROOT/benchmarks/pouw/harness/sass_gate.py" \
  "$OUT/units.sm_90a.cubin" --json "$OUT/harness-gate.json" > "$OUT/harness-gate.txt" 2>&1
echo "harness sass_gate.py rc $?" >> "$OUT/harness-gate.txt"
set -e
tail -3 "$OUT/harness-gate.txt"
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o units.sm_90a.cubin units.cu"; } > "$OUT/toolchain.txt"
cp "$HERE"/{words,lib,rmsnorm,quant,softmax,argmax,relu,add,sweep,bench,check}.py "$OUT/"
{ git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"
  git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/units verity 2>/dev/null || true; } > "$OUT/source.txt"
if [ -z "${REPLAY:-}" ] && [ -f "${RESEARCH_RUN_DIR:-/nonexistent}/inputs/replay.py" ]; then
  REPLAY=$RESEARCH_RUN_DIR/inputs/replay.py
fi
if [ -z "${ROWS:-}" ] && [ -f "${RESEARCH_RUN_DIR:-/nonexistent}/inputs/rows.py" ]; then
  ROWS=$RESEARCH_RUN_DIR/inputs/rows.py
fi
if [ -z "${CUBIN_ONLY:-}" ]; then
  (cd "$ROOT" && uv run python "$HERE/fixture.py" "$OUT/check" ${REPLAY:+--replay "$REPLAY"} ${ROWS:+--rows "$ROWS"})
fi
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "ship:"
