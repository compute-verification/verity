#!/usr/bin/env bash
# memory-accounting's P2 decode library for the H200, built off the card (a CPU node):
#
#   research run --on auto --cwd source -- bash benchmarks/torch_gpu/p2_stage/build.sh
#
# P2's prover's libp2v1.so (p2_store's), unchanged, by its own Makefile (FAST's default: v4 keys by p2keys4.cuh, p2sq.cuh's
# Karatsuba squaring), at ARCH (default sm_90a) with CGBN at p2_gpu/CGBN.commit. Writes into OUT (default $OUT):
#   lib/libp2v1.so, lib/libp2v1.so.ptxas   the library and ptxas's report (registers, spills)
#   lib/libp2v1.so.cubins/                 the library's cubins (ship.sh's ship_cubins)
#   elf.txt                                cuobjdump --list-elf: the SASS targets the library holds
#   toolchain.txt                          nvcc --version and the make line
#   source.txt                             the commit, and the sha256 of every source the Makefile names
#   MANIFEST.sha256                        every file above
# CUDA names a toolkit (nvcc >= 12.8); else the node's /usr/local/cuda-13.0 or /usr/local/cuda. It builds in ship.sh's
# frame from its loaded sources, with ship.sh's SHIP_NVCC as the Makefile's NVCC, so a rebuild at the same commit and
# toolkit path reproduces the cubins, and so does one after a source or a header only moved (ship_nvcc.sh's header).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../ship.sh"
OUT=${1:-${OUT:?usage: build.sh OUT}}; ARCH=${ARCH:-sm_90a}
P2=$(dirname verity/protocols/pous/prover/p2/p2_store/Makefile)
ship_stage "$P2" "$(dirname verity/protocols/pous/prover/p2/p2_gpu/p2dec.cu)" verity/circuits/kernels
P2=$SRC/$P2
S=$(mktemp -d "${TMPDIR:-/tmp}/p2-stage-build.XXXXXX"); trap 'rm -rf "$S"' EXIT
mkdir -p "$OUT/lib"

ship_cuda /usr/local/cuda-13.0 /usr/local/cuda

GMPINC=
if ! echo '#include <gmp.h>' | cpp >/dev/null 2>&1; then
  uv pip install -q --target "$S/py" gmpy2
  GMPINC=$(find "$S/py" -name gmp.h -printf '%h\n' | head -1)
  [ -n "$GMPINC" ] || { echo "gmpy2's wheel has no gmp.h" >&2; exit 2; }
fi
git clone -q https://github.com/NVlabs/CGBN "$S/CGBN"
git -C "$S/CGBN" checkout -q "$(cat "$ROOT/verity/protocols/pous/prover/p2/p2_gpu/CGBN.commit")"

mk=(make -C "$P2" NVCC="$SHIP_NVCC" CGBN="$S/CGBN" BIN="$OUT/lib" ARCH="$ARCH" ${GMPINC:+"INC=-I $GMPINC"})
{ "$CUDA/bin/nvcc" --version; echo "${mk[*]}"; } > "$OUT/toolchain.txt"
"${mk[@]}" > "$S/make.log" 2>&1 || { tail -n 40 "$S/make.log" "$OUT/lib/libp2v1.so.ptxas" >&2; exit 1; }
"$CUDA/bin/cuobjdump" --list-elf "$OUT/lib/libp2v1.so" > "$OUT/elf.txt"
grep -q "$ARCH" "$OUT/elf.txt" || { echo "libp2v1.so holds no $ARCH SASS" >&2; cat "$OUT/elf.txt" >&2; exit 1; }
ship_cubins "$OUT/lib/libp2v1.so"
if grep -E "spill (stores|loads)" "$OUT/lib/libp2v1.so.ptxas" | grep -vq " 0 bytes spill stores, 0 bytes spill loads"; then
  echo "ptxas reports spills (lib/libp2v1.so.ptxas)" >&2
fi
{ git -C "$ROOT" rev-parse HEAD
  (cd "$ROOT" && sha256sum verity/protocols/pous/prover/p2/p2_store/p2v1.cu verity/protocols/pous/prover/p2/p2_store/p2sq.cuh verity/protocols/pous/prover/p2/p2_store/p2keys.cuh \
    verity/protocols/pous/prover/p2/p2_store/p2keys4.cuh verity/protocols/pous/prover/p2/p2_store/Makefile \
    verity/protocols/pous/prover/p2/p2_gpu/p2dec.cu verity/protocols/pous/prover/p2/p2_gpu/CGBN.commit verity/circuits/kernels/sha512.cuh); } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
cat "$OUT/elf.txt"
grep -E "registers|spill" "$OUT/lib/libp2v1.so.ptxas" | sort | uniq -c | head -20
