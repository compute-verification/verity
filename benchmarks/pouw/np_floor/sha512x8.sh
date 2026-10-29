#!/usr/bin/env bash
# sha512x8.c on a GPU node's host: the eight-lane AVX-512 SHA-512, alone and two sets interleaved, with the core's clock
# under 512-bit work, on every thread and on one thread a core. Writes $OUT/cpu.txt, sha512x8.sass-counts.txt,
# sha512x8-<threads>.json and load.txt (the host's load before and after, since other jobs share it).
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
load() { echo "== $1"; cat /proc/loadavg; ps -eo pcpu,pid,comm --sort=-pcpu | head -12; } >>"$OUT/load.txt"
load before
n=$(nproc)
OMP_PLACES=threads OMP_PROC_BIND=close "$WORK/sha512x8" "$n" "$SECS" | tee "$OUT/sha512x8-$n.json"
OMP_PLACES=cores OMP_PROC_BIND=spread "$WORK/sha512x8" "$((n / 2))" "$SECS" | tee "$OUT/sha512x8-$((n / 2)).json"
load after
