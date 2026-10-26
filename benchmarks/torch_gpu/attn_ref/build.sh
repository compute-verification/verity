#!/usr/bin/env bash
# attn_ref's ship, built off the card on a CPU node with the CUDA toolkit:
#   attn_ref.sm_90a.cubin   attn_ref.cu with nvcc >= 12.8 (-fmad=false, no FTZ or fast-math flag, no -lineinfo), refused
#                           unless sass_gate.py passes it (no ptxas warning, spill or local memory; the only floating-point
#                           instructions HMMA.16816.F32.BF16 and the cvt's F2FP)
#   sass.json, ptxas.txt, attn_ref.sm_90a.sass   the gate's record, ptxas's report, the SASS
#   check/                  fixture.py's pool and odd case, from the Definition (needs verity: run under uv)
#   attn_ref.py, check.py, toolchain.txt (nvcc --version and the command line), source.txt (the commit), MANIFEST.sha256
# Usage: bash benchmarks/torch_gpu/attn_ref/build.sh [OUT]   (default /tmp/attn-ref-ship; CUDA names the toolkit, default the
#        first of /usr/local/cuda, /usr/local/cuda-13.0 with nvcc; CUBIN_ONLY=1 stops after the gate and needs no uv)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
if [ -z "${CUDA:-}" ]; then
  for c in /usr/local/cuda /usr/local/cuda-13.0; do [ -x "$c/bin/nvcc" ] && CUDA=$c && break; done
fi
CUDA=${CUDA:-/usr/local/cuda}; OUT=${1:-/tmp/attn-ref-ship}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -Xptxas -v -cubin)
case " ${FLAGS[*]} " in *ftz*|*fast-math*|*fast_math*|*lineinfo*) echo "a refused flag: ${FLAGS[*]}" >&2; exit 1;; esac
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/attn_ref.sm_90a.cubin" "$HERE/attn_ref.cu" 2> "$OUT/ptxas.txt" \
  || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$OUT/attn_ref.sm_90a.cubin" > "$OUT/attn_ref.sm_90a.sass"
python3 "$HERE/sass_gate.py" "$OUT/ptxas.txt" "$OUT/attn_ref.sm_90a.sass" "$OUT/sass.json"
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o attn_ref.sm_90a.cubin attn_ref.cu"; } > "$OUT/toolchain.txt"
cp "$HERE/attn_ref.py" "$HERE/check.py" "$OUT/"
{ git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"
  git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/attn_ref verity 2>/dev/null || true; } > "$OUT/source.txt"
if [ -z "${CUBIN_ONLY:-}" ]; then
  (cd "$ROOT" && uv run python "$HERE/fixture.py" "$OUT/check")
fi
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "ship:"
