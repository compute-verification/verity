#!/usr/bin/env bash
# memory-accounting's P2 decode library for the H200, built off the card (a CPU node):
#
#   research run --on auto --cwd source -- bash benchmarks/torch_gpu/p2_stage/build.sh
#
# benchmarks/pous/p2_v1's libp2v1.so, unchanged, by its own Makefile (FAST's default: v4 keys by p2keys4.cuh, p2sq.cuh's
# Karatsuba squaring), at ARCH (default sm_90a) with CGBN at p2_gpu/CGBN.commit. Writes into OUT (default $OUT):
#   lib/libp2v1.so, lib/libp2v1.so.ptxas   the library and ptxas's report (registers, spills)
#   elf.txt                                cuobjdump --list-elf: the SASS targets the library holds
#   toolchain.txt                          nvcc --version and the make line
#   source.txt                             the commit, and the sha256 of every source the Makefile names
#   MANIFEST.sha256                        every file above
# CUDA names a toolkit (nvcc >= 12.8); else the node's /usr/local/cuda-13.0 or /usr/local/cuda.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${1:-${OUT:?usage: build.sh OUT}}; ARCH=${ARCH:-sm_90a}
P2=$ROOT/benchmarks/pous/p2_v1
S=$(mktemp -d "${TMPDIR:-/tmp}/p2-stage-build.XXXXXX"); trap 'rm -rf "$S"' EXIT
mkdir -p "$OUT/lib"

release_ok() { [ -x "$1/bin/nvcc" ] && "$1/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\."; }
CUDA=${CUDA:-}
if [ -z "$CUDA" ]; then
  for home in /usr/local/cuda-13.0 /usr/local/cuda; do release_ok "$home" && { CUDA=$home; break; }; done
fi
[ -n "$CUDA" ] && release_ok "$CUDA" || { echo "no nvcc >= 12.8 here (set CUDA)" >&2; exit 2; }

GMPINC=
if ! echo '#include <gmp.h>' | cpp >/dev/null 2>&1; then
  uv pip install -q --target "$S/py" gmpy2
  GMPINC=$(find "$S/py" -name gmp.h -printf '%h\n' | head -1)
  [ -n "$GMPINC" ] || { echo "gmpy2's wheel has no gmp.h" >&2; exit 2; }
fi
git clone -q https://github.com/NVlabs/CGBN "$S/CGBN"
git -C "$S/CGBN" checkout -q "$(cat "$ROOT/benchmarks/pous/p2_gpu/CGBN.commit")"

mk=(make -C "$P2" NVCC="$CUDA/bin/nvcc" CGBN="$S/CGBN" BIN="$OUT/lib" ARCH="$ARCH" ${GMPINC:+"INC=-I $GMPINC"})
{ "$CUDA/bin/nvcc" --version; echo "${mk[*]}"; } > "$OUT/toolchain.txt"
"${mk[@]}" > "$S/make.log" 2>&1 || { tail -40 "$S/make.log" "$OUT/lib/libp2v1.so.ptxas" >&2; exit 1; }
"$CUDA/bin/cuobjdump" --list-elf "$OUT/lib/libp2v1.so" > "$OUT/elf.txt"
grep -q "$ARCH" "$OUT/elf.txt" || { echo "libp2v1.so holds no $ARCH SASS" >&2; cat "$OUT/elf.txt" >&2; exit 1; }
if grep -E "spill (stores|loads)" "$OUT/lib/libp2v1.so.ptxas" | grep -vq " 0 bytes spill stores, 0 bytes spill loads"; then
  echo "ptxas reports spills (lib/libp2v1.so.ptxas)" >&2
fi
{ git -C "$ROOT" rev-parse HEAD
  (cd "$ROOT" && sha256sum benchmarks/pous/p2_v1/{p2v1.cu,p2sq.cuh,p2keys.cuh,p2keys4.cuh,Makefile} \
    benchmarks/pous/p2_gpu/{p2dec.cu,CGBN.commit}); } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
cat "$OUT/elf.txt"
grep -E "registers|spill" "$OUT/lib/libp2v1.so.ptxas" | sort | uniq -c | head -20
