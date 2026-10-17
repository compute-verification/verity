#!/usr/bin/env bash
# One word's squaring latency on one CPU core, at every Sloth width the codec table names (cwd = the source tree):
# lat.c (GMP's mpn_sqr and a fold by C, any width; unsigned __int128 at 256 and 512 bits) and coop.c (AVX-512 IFMA, one
# word's 52-bit limbs across the lanes, on T cooperating cores of one L3), every chain checked against mpz_powm, and the
# core's clock off a dependent imul chain before and after. JSON lines in $OUT. Pin it to the cores of one L3:
#
#   research run --on vy-nebius-2 --cwd source -- taskset -c 48-55 bash benchmarks/codec_cost/cpu_lat.sh
set -uxo pipefail
O=${OUT:?}
D=benchmarks/codec_cost
B=$(mktemp -d /tmp/codec-cost.XXXXXX); trap 'rm -rf "$B"' EXIT
if [ ! -e /usr/include/gmp.h ] && [ ! -e /usr/include/x86_64-linux-gnu/gmp.h ]; then
  sudo -n apt-get install -y libgmp-dev > "$O/apt.log" 2>&1
fi
lscpu > "$O/lscpu.txt"
lscpu -p=CPU,CORE,SOCKET,CACHE > "$O/lscpu-p.txt"
grep -o -m1 avx512ifma /proc/cpuinfo > "$O/ifma.txt" || echo none > "$O/ifma.txt"
# one allowed CPU a physical core, all under the L3 of the first allowed CPU
COOP_CPUS=$(python3 - "$O/lscpu-p.txt" <<'EOF'
import os, sys
allowed = sorted(os.sched_getaffinity(0))
rows = {}
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    cpu, core, sock, cache = line.strip().split(",")
    rows[int(cpu)] = (core, sock, cache.split(":")[-1])
l3 = rows[allowed[0]][1:]
cores, out = set(), []
for c in allowed:
    if rows[c][1:] == l3 and rows[c][0] not in cores:
        cores.add(rows[c][0])
        out.append(str(c))
print(",".join(out[:8]))
EOF
)
export COOP_CPUS
echo "$COOP_CPUS" > "$O/coop-cpus.txt"
WIDTHS=${WIDTHS:-"127:1 255:19 256:189 511:481 512:569 521:1 767:1081 768:825 1023:361 1024:105 1279:1 1536:3453 2048:1557 2112:2397 4160:1017 8256:7305 16448:21065"}
for wc in $WIDTHS; do
  w=${wc%%:*}; c=${wc##*:}
  gcc -O3 -march=native -DW="$w" -DC="$c" -o "$B/lat$w" $D/lat.c -lgmp 2>> "$O/build.log" || echo "lat $w build rc $?" >> "$O/errors.txt"
  gcc -O3 -march=native -pthread -DW="$w" -DC="$c" -o "$B/coop$w" $D/coop.c -lgmp 2>> "$O/build.log" || echo "coop $w build rc $?" >> "$O/errors.txt"
done
"$B/lat512" clock >> "$O/clock.jsonl"
for wc in $WIDTHS; do
  w=${wc%%:*}
  n=$(( 40000000 / w )); [ $n -lt 2000 ] && n=2000
  "$B/lat$w" chain 9 $n >> "$O/mpn.jsonl" 2>> "$O/errors.txt" || echo "lat $w rc $?" >> "$O/errors.txt"
  if [ "$w" = 256 ] || [ "$w" = 512 ]; then
    "$B/lat$w" u128 9 $n >> "$O/u128.jsonl" 2>> "$O/errors.txt" || echo "u128 $w rc $?" >> "$O/errors.txt"
  fi
  "$B/coop$w" ifma 1 9 $n >> "$O/ifma.jsonl" 2>> "$O/errors.txt" || echo "coop $w T1 rc $?" >> "$O/errors.txt"
  if [ "$w" -ge 2048 ]; then
    for t in 2 4 $(( $(tr -cd , <<< "$COOP_CPUS" | wc -c) + 1 )); do
      "$B/coop$w" ifma $t 9 $n >> "$O/ifma.jsonl" 2>> "$O/errors.txt" || echo "coop $w T$t rc $?" >> "$O/errors.txt"
      if [ "$w" -ge 8256 ]; then
        "$B/coop$w" owner $t 9 $n >> "$O/owner.jsonl" 2>> "$O/errors.txt" || echo "owner $w T$t rc $?" >> "$O/errors.txt"
      fi
    done
  fi
done
"$B/lat512" clock >> "$O/clock.jsonl"
"$B/coop512" pingpong 2 5 >> "$O/pingpong.jsonl" 2>> "$O/errors.txt" || true
