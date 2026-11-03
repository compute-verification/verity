#!/usr/bin/env bash
# ε_skip's per-instruction counts off sm_120 (eps_budget.py's APV_TAIL, c_apv_serve and c_vfold, the checked kernels'
# and the D2S kernels' ncu counts on sm_120): build the kernels they came from (edited_gemm/apv.cu: apv_l7qs, apv_cc,
# apv_plain, vfold, vfold_x) for sm_120a, sm_90a and sm_100a with -lineinfo, and dump each cubin's SASS with its inlined
# line info for eps_counts.py, which counts each kernel's tail, served epilogue and sums by the source lines they come
# from. CPU only, no GPU. Each source tree is a `git archive` of apv.cu's include closure at the commit its counts were
# taken at (benchmarks/pouw/{edited_gemm,masked_gemm,floored_gemm,pearl_c}), sent as src-TAG.tar:
#
#   git archive -o src-91357d734.tar 91357d734 benchmarks/pouw/{edited_gemm,masked_gemm,floored_gemm,pearl_c}
#   research run --on MACHINE --project verity --source <tree> --cwd source --mem-gb 16 \
#     --send src-91357d734.tar ... -- bash benchmarks/pouw/sass_census/eps_counts.sh
#
# Writes under $OUT/TAG/ARCH: build.log (ptxas -v), apv.gi.sass.gz (nvdisasm -c -gi), apv.sass.gz (cuobjdump -sass), and
# lineinfo.txt: whether the -lineinfo cubin's SASS is the plain build's, instruction for instruction.
set -uo pipefail
OUT=${OUT:?run under research run}
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "eps_counts.sh: no nvcc" >&2; exit 2; }
BIN=$(dirname "$NVCC")
ARCHS=${ARCHS:-"sm_120a sm_90a sm_100a"}
JOBS=${JOBS:-6}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
{ "$NVCC" --version; "$BIN/nvdisasm" --version; } >"$OUT/toolkit.txt" 2>&1
tags=()
for t in "$IN"/src-*.tar; do
  [ -e "$t" ] || { echo "eps_counts.sh: no src-*.tar in $IN" >&2; exit 2; }
  tag=$(basename "$t" .tar); tag=${tag#src-}
  mkdir -p "$WORK/$tag" && tar -xf "$t" -C "$WORK/$tag" && tags+=("$tag")
done

build() {
  local tag=$1 arch=$2 src="$WORK/$1/benchmarks/pouw/edited_gemm/apv.cu" o="$OUT/$1/$2" c="$WORK/$1-$2"
  mkdir -p "$o"
  local flags=(-O3 -std=c++17 --no-compress -gencode "arch=compute_${arch#sm_},code=$arch" -cubin)
  "$NVCC" "${flags[@]}" -lineinfo -Xptxas -v -o "$c.cubin" "$src" >"$o/build.log" 2>&1 || {
    echo "build $tag $arch rc $?" | tee -a "$o/build.log" >&2; return 1; }
  "$NVCC" "${flags[@]}" -o "$c.nl.cubin" "$src" >"$o/build.nl.log" 2>&1 || {
    echo "build (no lineinfo) $tag $arch rc $?" | tee -a "$o/build.nl.log" >&2; return 1; }
  "$BIN/nvdisasm" -c -gi "$c.cubin" | gzip >"$o/apv.gi.sass.gz"
  "$BIN/cuobjdump" -sass "$c.cubin" >"$c.sass"
  "$BIN/cuobjdump" -sass "$c.nl.cubin" >"$c.nl.sass"
  gzip -c "$c.sass" >"$o/apv.sass.gz"
  if cmp -s <(grep -E '^\s+/\*[0-9a-f]{4,}\*/' "$c.sass") <(grep -E '^\s+/\*[0-9a-f]{4,}\*/' "$c.nl.sass"); then
    echo "same: -lineinfo leaves the SASS as the plain build's" >"$o/lineinfo.txt"
  else
    echo "differs: -lineinfo changed the SASS" >"$o/lineinfo.txt"
  fi
  echo "built $tag $arch"
}
export -f build
export NVCC BIN WORK OUT
for tag in "${tags[@]}"; do for a in $ARCHS; do echo "$tag $a"; done; done |
  xargs -P "$JOBS" -n 2 bash -c 'build "$0" "$1"'
rc=$?
find "$OUT" -name lineinfo.txt -exec grep -H . {} + | sed "s|$OUT/||"
exit $rc
