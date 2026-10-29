#!/usr/bin/env bash
# The host CPU's rates (host.c) on the node it runs on: one thread for SECS seconds a test, then every thread for
# ALL_SECS (0 skips it); OpenSSL's SHA-512 and sha512x8.c's eight-lane AVX-512 SHA-512 on 1, half and every thread.
# Writes $OUT/cpu.txt (lscpu), $OUT/host-<threads>.json, $OUT/sha512-<threads>.txt and $OUT/sha512x8*.json.
#   research run --on NODE --project verity --campaign pouw-night --source <tree> --cwd source -- \
#     bash benchmarks/pouw/np_floor/host.sh [SECS] [ALL_SECS]
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SECS=${1:-1}
ALL_SECS=${2:-0.25}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
lscpu >"$OUT/cpu.txt"
gcc --version | head -1 >"$OUT/gcc.txt"
openssl version >>"$OUT/gcc.txt" 2>&1 || true
gcc -O3 -march=sapphirerapids -fopenmp "$HERE/host.c" -o "$WORK/host"
objdump -d "$WORK/host" | grep -c -E 'tdpbssd' | xargs echo "tdpbssd sites:" >>"$OUT/gcc.txt"
objdump -d "$WORK/host" | grep -c -E 'vpdpbusd' | xargs echo "vpdpbusd sites:" >>"$OUT/gcc.txt"
"$WORK/host" 1 "$SECS" | tee "$OUT/host-1.json"
s=${SECS%.*}
[ "${s:-0}" -ge 1 ] || s=1
openssl speed -seconds "$s" -bytes 16384 -evp sha512 2>/dev/null | tail -3 | tee "$OUT/sha512-1.txt"
openssl speed -seconds "$s" -bytes 262144 -evp sha512 2>/dev/null | tail -1 | tee -a "$OUT/sha512-1.txt"
gcc -O3 -march=sapphirerapids -fopenmp "$HERE/sha512x8.c" -o "$WORK/sha512x8"
objdump -d "$WORK/sha512x8" | grep -c -E 'vprorq' | xargs echo "sha512x8 vprorq sites:" >>"$OUT/gcc.txt"
objdump -d "$WORK/sha512x8" | grep -c -E 'vpternlogq' | xargs echo "sha512x8 vpternlogq sites:" >>"$OUT/gcc.txt"
if [ "$ALL_SECS" != 0 ]; then
  n=$(nproc)
  "$WORK/host" "$n" "$ALL_SECS" | tee "$OUT/host-$n.json"
  openssl speed -multi "$n" -seconds 1 -bytes 16384 -evp sha512 2>/dev/null | tail -3 | tee "$OUT/sha512-$n.txt"
  "$WORK/sha512x8" "$n" "$SECS" | tee "$OUT/sha512x8.json"
  "$WORK/sha512x8" "$((n / 2))" "$SECS" | tee "$OUT/sha512x8-half.json"
else
  "$WORK/sha512x8" 1 "$SECS" | tee "$OUT/sha512x8.json"
fi
