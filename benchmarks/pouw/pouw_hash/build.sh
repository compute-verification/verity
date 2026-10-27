#!/usr/bin/env bash
# pouw_hash's shipped tree, built off the device:
#   pouw_hash_bench   pouw_hash_bench.cu (timed kernels) and pouw_hash_check.cu (check kernels) with CUDA 12.9.1's nvcc for
#                     sm_120a (cudart static; NVML from the driver at run time), about half a minute,
#                     refused unless sass.py's gates pass (no local memory in a timed kernel, each op kernel's opcode, the
#                     epilogue's mma.sync counts) and it needs no glibc newer than 2.35
#   host_check.jsonl  the vectors from the host build (g++ -x c++ pouw_hash_check.cu), refused unless check.py holds every row
#                     to its reference
#   sass.json         registers, local memory and the opcode census per kernel; ptxas.txt
#   run.sh, check.py, the sources, toolchain.txt, source.txt (the verity commit), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pouw_hash/build.sh [OUT]   (default /tmp/pouw-hash-ship; C9 names CUDA 12.9, default
# /tmp/cuda-12.9.1 as the store's code/pouw-gpu-constants/h100/fetch_cuda.py 12.9.1 lays it out, or /usr/local/cuda-12.9). Ship it with benchmarks/pouw/pouw_hash/ship.sh OUT REPO, then `research run --source REPO --cwd source`.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
C9=${C9:-/tmp/cuda-12.9.1}; OUT=${1:-/tmp/pouw-hash-ship}
[ -x "$C9/bin/nvcc" ] || { echo "no nvcc at $C9 (CUDA 12.9.1)" >&2; exit 1; }
"$C9/bin/nvcc" --version | grep -q "release 12.9" || { echo "$C9 is not CUDA 12.9" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
g++ -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off -x c++ -o "$OUT/pouw_hash_host" "$HERE/pouw_hash_check.cu"
"$OUT/pouw_hash_host" check > "$OUT/host_check.jsonl"
(cd "$ROOT" && uv run python "$HERE/check.py" "$OUT/host_check.jsonl")
rm "$OUT/pouw_hash_host"
NVCC=("$C9/bin/nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a -Xptxas -v -I "$HERE"
      -I "$C9/nvml/include" -I "$C9/include")   # nvml.h: the redist's nvml/ (fetch_cuda.py) or an apt install's include/
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
"${NVCC[@]}" -c -o "$TMP/check.o" "$HERE/pouw_hash_check.cu" 2> "$TMP/check.txt" &
"${NVCC[@]}" -c -o "$TMP/bench.o" "$HERE/pouw_hash_bench.cu" 2> "$TMP/bench.txt" || { cat "$TMP/bench.txt" >&2; exit 1; }
wait $! || { cat "$TMP/check.txt" >&2; exit 1; }
cat "$TMP/check.txt" "$TMP/bench.txt" > "$OUT/ptxas.txt"
"$C9/bin/nvcc" -gencode arch=compute_120a,code=sm_120a -L "$C9/lib" -L "$C9/lib/stubs" -L "$C9/nvml/lib/stubs" -L "$C9/lib64/stubs" \
  -o "$OUT/pouw_hash_bench" "$TMP/check.o" "$TMP/bench.o" -lnvidia-ml
"$C9/bin/cuobjdump" -sass "$OUT/pouw_hash_bench" > "$OUT/sass.txt"
python3 "$HERE/sass.py" "$OUT/ptxas.txt" "$OUT/sass.txt" "$OUT/sass.json" | c++filt
rm "$OUT/sass.txt"
NEW=$(objdump -T "$OUT/pouw_hash_bench" | grep -oE "GLIBC_2\.[0-9]+" | sort -uV | awk -F. '$2 > 35' | tr '\n' ' ')
[ -z "$NEW" ] || { echo "REFUSED: pouw_hash_bench needs $NEW (newer than 2.35)" >&2; exit 1; }
cp "$HERE/run.sh" "$HERE/check.py" "$HERE/pouw_hash.cuh" "$HERE/vectors.cuh" "$HERE/pouw_hash_check.cu" "$HERE/pouw_hash_bench.cu" "$OUT/"
# every file 0644, as ship.sh commits them, so the shipped repo's checkout is clean for `research run --source`;
# run.sh makes the binary executable on the machine
chmod 0644 "$OUT/pouw_hash_bench"
"$C9/bin/nvcc" --version | tail -2 > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pouw_hash; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
