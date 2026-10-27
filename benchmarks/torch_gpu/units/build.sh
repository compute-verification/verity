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
#                         names the CPU replay for fixture.py's cross-check (default the tree's examples/h200/replay.py),
#                         ROWS an ops/rows.py to load in place of the installed one
#   the unit modules, lib.py, bench.py, check.py, settings.py (their registry, which they find beside them),
#   toolchain.txt (nvcc --version and the command lines), source.txt (the commit), MANIFEST.sha256
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
"$CUDA/bin/nvcc" --version > "$OUT/toolchain.txt"
for A in 90a 120a; do
  FLAGS=(-O3 -std=c++17 -gencode "arch=compute_$A,code=sm_$A" -fmad=false -Xptxas -v -cubin)
  case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*|*lineinfo*|*prec-div*|*prec-sqrt*) echo "a refused flag: ${FLAGS[*]}" >&2; exit 1;; esac
  "$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/units.sm_$A.cubin" "$HERE/units.cu" 2> "$OUT/ptxas.sm_$A.txt" \
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
{ git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"
  git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/units verity 2>/dev/null || true; } > "$OUT/source.txt"
REPLAY=${REPLAY:-$ROOT/examples/h200/replay.py}
if [ -z "${CUBIN_ONLY:-}" ]; then
  (cd "$ROOT" && uv run python "$HERE/fixture.py" "$OUT/check" --replay "$REPLAY" ${ROWS:+--rows "$ROWS"})
fi
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "ship:"
