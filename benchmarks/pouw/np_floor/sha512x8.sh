#!/usr/bin/env bash
# sha512x8.c on a GPU node's host: the eight-lane AVX-512 SHA-512, alone and two sets interleaved, with the core's clock
# under 512-bit work, on every thread and on half of them. Writes $OUT/cpu.txt, sha512x8.sass-counts.txt and
# sha512x8-<threads>.json.
#   research run --on vy-dc-h200-5 --project verity --campaign pouw-night --source <tree> --cwd source \
#     -- bash benchmarks/pouw/np_floor/sha512x8.sh [SECONDS]
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SECS=${1:-1}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
lscpu >"$OUT/cpu.txt"
gcc -O3 -march=sapphirerapids -fopenmp "$HERE/sha512x8.c" -o "$WORK/sha512x8"
for op in vprorq vpternlogq vpaddq vpsrlq; do
  echo "$op $(objdump -d --no-show-raw-insn "$WORK/sha512x8" | grep -c -w "$op")"
done >"$OUT/sha512x8.sass-counts.txt"
n=$(nproc)
for t in "$n" "$((n / 2))"; do "$WORK/sha512x8" "$t" "$SECS" | tee "$OUT/sha512x8-$t.json"; done
