#!/usr/bin/env bash
# The SASS opcode census of a node's binaries: every file with device code under the given roots (the CUDA toolkit
# libraries, NCCL, the node's wheel/kernel caches, the repo's own compiled kernels), disassembled once with
# `cuobjdump -sass` (one pass dumps every embedded architecture), its opcodes counted by `sass_count.py`. CPU only;
# no GPU. Writes under $OUT:
#   toolkit.txt    cuobjdump and nvcc --version
#   corpus.tsv     a line a binary kept: sha256, bytes, the architectures it holds, its path
#   counts/*.json  `sass_count.py count` for each binary (every architecture it holds)
#   census.json    `sass_count.py merge` of them all, a device family each; summary.txt its totals
#
#   research run --on MACHINE --cwd source -- bash benchmarks/pouw/sass_census/dump.sh [ROOT ...]
# MAXBYTES skips binaries larger than it: a fast first pass that leaves out cuBLASLt (whose one cuobjdump pass is serial
# and holds the merge until it ends), for a first count while the full dump runs.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "$0")" && pwd)
NVCC=${NVCC:-$(ls -d /usr/local/cuda-13*/bin/nvcc 2>/dev/null | sort -V | tail -1)}
[ -x "${NVCC:-}" ] || NVCC=$(command -v nvcc)
BIN=$(dirname "$NVCC")
CUOBJDUMP=${CUOBJDUMP:-$BIN/cuobjdump}
JOBS=${JOBS:-$(nproc)}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/counts"
{ "$CUOBJDUMP" --version; "$NVCC" --version; } >"$OUT/toolkit.txt" 2>&1

roots=("$@")
[ ${#roots[@]} -gt 0 ] || roots=(/usr/local/cuda /usr/lib/x86_64-linux-gnu /workspace/cache /home/research/.cache \
  /home/research/.triton "$HERE/../../../")
for r in "${roots[@]}"; do
  [ -e "$r" ] || continue
  find -L "$r" -xdev -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.cubin' -o -name '*.fatbin' \) \
    -size +100k 2>/dev/null
done | sort -u >"$WORK/candidates"

# One copy of each binary (the same wheel sits in several venvs), and only binaries with device code.
: >"$OUT/corpus.tsv"
while read -r f; do
  [ -z "${MAXBYTES:-}" ] || [ "$(stat -c %s "$f")" -le "$MAXBYTES" ] || continue
  archs=$("$CUOBJDUMP" -lelf "$f" 2>/dev/null | grep -o 'sm_[0-9]*[a-z]\?' | sort -u | tr '\n' ' ')
  [ -n "$archs" ] || archs=$("$CUOBJDUMP" -sass "$f" 2>/dev/null | grep -m1 -o 'sm_[0-9]*[a-z]\?')
  [ -n "$archs" ] || continue
  printf '%s\t%s\t%s\t%s\n' "$(sha256sum "$f" | cut -c1-16)" "$(stat -c %s "$f")" "$archs" "$f"
done <"$WORK/candidates" | sort -t$'\t' -k1,1 -u >"$OUT/corpus.tsv"
wc -l <"$OUT/corpus.tsv" >"$OUT/corpus.count"

export CUOBJDUMP HERE OUT
work() {
  IFS=$'\t' read -r sha bytes archs f <<<"$1"
  "$CUOBJDUMP" -sass "$f" 2>/dev/null \
    | python3 "$HERE/sass_count.py" count --source "$(basename "$f")@$sha" >"$OUT/counts/${sha}.json"
}
export -f work
# the largest binaries first, so the long ones start early; one cuobjdump pass a binary
sort -t$'\t' -k2,2nr "$OUT/corpus.tsv" | tr '\n' '\0' | xargs -0 -P "$JOBS" -I{} bash -c 'work "$@"' _ {}

python3 "$HERE/sass_count.py" merge "$OUT"/counts/*.json >"$OUT/census.json"
python3 "$HERE/sass_count.py" summary "$OUT/census.json" | tee "$OUT/summary.txt"
