#!/usr/bin/env bash
# Writes zstd_entry.fatbin, the fatbin test_sass_gate.py reads a Zstandard entry from: the .nv_fatbin section of
# zstd_entry.cu's object, built for sm_90 by CUDA 13.3's nvcc with --compress-mode=size (its default leaves a cubin this
# small as it is, and compresses a large one into the same entry: flag 0x8000, one Zstandard frame). Prints the sha256
# of the cubin `cuobjdump -xelf` unpacks from the object, which is the test's ZSTD_CUBIN_SHA256, and fails unless the
# same nvcc's --no-compress build holds that cubin byte for byte.
#
#   CUDA_HOME=/usr/local/cuda-13.3 benchmarks/pouw/tests/fixtures/sass_gate/zstd_entry.sh [OUT]
#
# Needs nvcc (CUDA_HOME/bin), cuobjdump (POUW_CUDA_BIN, else beside nvcc) and objcopy. OUT defaults to the fixture.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=$(realpath "${1:-$here/zstd_entry.fatbin}")
nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
cuobjdump=${POUW_CUDA_BIN:-$(dirname "$nvcc")}/cuobjdump
"$nvcc" --version | grep -q 'release 13\.3' || { echo "zstd_entry.sh: the fixture is CUDA 13.3's; $nvcc is $("$nvcc" --version | tail -1)" >&2; exit 2; }
w=$(mktemp -d); trap 'rm -rf "$w"' EXIT
mkdir "$w/size" "$w/plain"
build() { "$nvcc" -O3 -std=c++17 -gencode arch=compute_90,code=sm_90 "$1" -c "$here/zstd_entry.cu" -o "$w/$2/k.o"; }
build --compress-mode=size size
build --no-compress plain
for d in size plain; do (cd "$w/$d" && "$cuobjdump" -xelf all k.o > /dev/null && sha256sum -- *.cubin | cut -d' ' -f1 > sha); done
[ "$(wc -l < "$w/size/sha")" = 1 ] && cmp -s "$w/size/sha" "$w/plain/sha" \
  || { echo "zstd_entry.sh: the compressed build's cubin is not the --no-compress build's" >&2; exit 1; }
objcopy -O binary --only-section=.nv_fatbin "$w/size/k.o" "$out"
echo "cubin sha256 $(cat "$w/size/sha")"
echo "fatbin sha256 $(sha256sum "$out" | cut -d' ' -f1), $(stat -c %s "$out") bytes"
