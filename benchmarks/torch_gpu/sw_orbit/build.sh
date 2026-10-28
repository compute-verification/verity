#!/usr/bin/env bash
# orbit.cu's library for sm_90a, and s8_linear's wgmma.linear cubin beside it, into OUT:
#   liborbit.so      orbit.cu with nvcc >= 12.8 against CUTLASS at the harness's pin (benchmarks/pouw/harness/native/build.sh's
#                    CUTLASS_TAG and CUTLASS_COMMIT): the checkout in CUTLASS_CACHE (default /workspace/pouw/accel_v31's, which
#                    accel_v31's run.sh keeps) when it's at the pin, else a fresh clone under OUT, removed after the build
#   ptxas.txt        ptxas -v's registers, spills and shared memory per kernel
#   s8_wgmma.cubin   s8_linear's kernel as its build.sh compiles it (no SASS gate here: wgmma.py reads it as a speed baseline)
#   toolchain.txt, source.txt
# Usage: bash benchmarks/torch_gpu/sw_orbit/build.sh OUT   (CUDA names the toolkit, default /usr/local/cuda-13.0)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CUDA=${CUDA:-/usr/local/cuda-13.0}; OUT=${1:?OUT}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA: set CUDA" >&2; exit 1; }
H=$ROOT/benchmarks/pouw/harness/native/build.sh
TAG=$(sed -n 's/^CUTLASS_TAG="\(.*\)"$/\1/p' "$H"); PIN=$(sed -n 's/^CUTLASS_COMMIT="\(.*\)"$/\1/p' "$H")
mkdir -p "$OUT"
CUT=${CUTLASS_CACHE:-/workspace/pouw/accel_v31}/cutlass-$TAG
if [ "$(git -C "$CUT" rev-parse HEAD 2>/dev/null)" != "$PIN" ]; then
  CUT=$OUT/cutlass; rm -rf "$CUT"
  git clone -q --depth 1 --branch "$TAG" https://github.com/NVIDIA/cutlass "$CUT"
  [ "$(git -C "$CUT" rev-parse HEAD)" = "$PIN" ] || { echo "CUTLASS $TAG is not at $PIN" >&2; exit 1; }
  trap 'rm -rf "$OUT/cutlass"' EXIT
fi
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -DNDEBUG
       -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -I"$CUT/include" -I"$CUT/tools/util/include" -Xptxas -v)
"$CUDA/bin/nvcc" "${FLAGS[@]}" -Xcompiler -fPIC -shared -o "$OUT/liborbit.so" "$HERE/orbit.cu" -lcuda \
  2> "$OUT/ptxas.txt" || { tail -60 "$OUT/ptxas.txt" >&2; exit 1; }
S8=$ROOT/benchmarks/torch_gpu/s8_linear
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -cubin -o "$OUT/s8_wgmma.cubin" \
  "$S8/s8_wgmma.cu" 2> "$OUT/ptxas-s8_wgmma.txt" || { tail -30 "$OUT/ptxas-s8_wgmma.txt" >&2; exit 1; }
{ "$CUDA/bin/nvcc" --version; echo "CUTLASS $TAG $PIN at $CUT"; echo "nvcc ${FLAGS[*]} -shared -o liborbit.so orbit.cu"; } \
  > "$OUT/toolchain.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/source.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/source.txt"
grep -E "Compiling entry|registers|spill" "$OUT/ptxas.txt" | grep -B1 -E "orbit_fused|forms|assemble" | head -20 || true
