#!/usr/bin/env bash
# SHA-512 throughput per core on this host's CPU (cpu_sha512.c): builds nasm and isa-l_crypto (both pinned by sha256) under CACHE,
# then times isa-l_crypto's multi-buffer managers and OpenSSL over LENS at 1 thread, one per physical core and one per CPU, plus
# `openssl speed`. Feasibility of hashing the Commit's values on the host after a copy-engine D2H; measurement only.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
CACHE=${CACHE:-$HOME/.cache/verity-cpu-sha512}
LENS=${LENS:-80,273,1024,4113,65536}
MIB=${MIB:-16}
REPS=${REPS:-10}
mkdir -p "$OUT" "$CACHE"
HERE=$(cd "$(dirname "$0")" && pwd)

fetch() {  # url sha256 dest
  [ -f "$3" ] || { curl -fsSL -o "$3.part" "$1" && mv "$3.part" "$3"; }
  echo "$2  $3" | sha256sum -c --quiet
}
NASM=nasm-2.16.03
ISAL=isa-l_crypto-2.25.0
fetch "https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/$NASM.tar.xz" \
  1412a1c760bbd05db026b6c0d1657affd6631cd0a63cddb6f73cc6d4aa616148 "$CACHE/$NASM.tar.xz"
fetch "https://github.com/intel/isa-l_crypto/archive/refs/tags/v2.25.0.tar.gz" \
  afe013e8eca17c9a0e567709c6a967f6b2b497d5317914afc98d5969599ee87e "$CACHE/$ISAL.tar.gz"
if [ ! -x "$CACHE/nasm/bin/nasm" ]; then
  (cd "$CACHE" && rm -rf "$NASM" && tar xJf "$NASM.tar.xz" && cd "$NASM" && ./configure -q --prefix="$CACHE/nasm" &&
    make -s -j"$(nproc)" install) > "$OUT/nasm_build.log" 2>&1
fi
if [ ! -f "$CACHE/$ISAL/bin/isa-l_crypto.a" ]; then
  (cd "$CACHE" && rm -rf "$ISAL" && tar xzf "$ISAL.tar.gz" && cd "$ISAL" &&
    make -f Makefile.unx -j"$(nproc)" AS="$CACHE/nasm/bin/nasm" lib) > "$OUT/isal_build.log" 2>&1
fi
gcc -O3 -march=native -Wno-deprecated-declarations -I"$CACHE/$ISAL/include" -o "$CACHE/cpu_sha512" "$HERE/cpu_sha512.c" \
  "$CACHE/$ISAL/bin/isa-l_crypto.a" -l:libcrypto.so.3 -lpthread

CPUS=$(nproc)
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
{
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket|MHz'
  echo "avx512: $(grep -o -w -E 'avx512f|avx512vl|avx512bw' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  uptime
} | tee "$OUT/cpu.txt"
THREADS=${THREADS:-$(printf '%s\n' 1 "$CORES" "$CPUS" | sort -nu | paste -sd,)}
"$CACHE/cpu_sha512" "${IMPLS:-isal-avx512,isal-avx2,openssl}" "$LENS" "$THREADS" "$MIB" "$REPS" | tee "$OUT/cpu_sha512.jsonl"
openssl speed -seconds 2 -bytes 4096 -evp sha512 2>/dev/null | tail -n 2 | tee "$OUT/openssl_speed.txt"
openssl speed -seconds 2 -bytes 4096 -multi "$CPUS" -evp sha512 2>/dev/null | tail -n 2 | tee -a "$OUT/openssl_speed.txt"
