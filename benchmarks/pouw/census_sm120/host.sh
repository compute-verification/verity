#!/usr/bin/env bash
# Host section of census.sh: host.c's STREAM-like bandwidth, AMX tile throughput and disk sequential read, all under
# nice 19 with at most 16 threads, pinned away from the CPUs the fill jobs use (96-127 on vy-nebius-2), plus lscpu,
# lsblk, NUMA layout and memory size. Usage: host.sh WORKDIR (a scratch directory census.sh deletes); writes
# $OUT/host.jsonl and $OUT/host-*.txt.
set -uo pipefail
OUT=${OUT:?}
W=${1:?workdir}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lscpu >"$OUT/host-lscpu.txt" 2>&1
lsblk -o NAME,SIZE,TYPE,ROTA,MODEL,MOUNTPOINT >"$OUT/host-lsblk.txt" 2>&1
free -b >"$OUT/host-free.txt" 2>&1
(numactl -H || cat /sys/devices/system/node/node*/meminfo) >"$OUT/host-numa.txt" 2>&1
df -h /tmp /workspace >"$OUT/host-df.txt" 2>&1
gcc -O3 -march=native -fopenmp -mamx-tile -mamx-int8 -mamx-bf16 -o "$W/host" "$SRC/host.c" -lpthread >"$OUT/build-host.log" 2>&1 \
  || { echo "host build failed" >&2; exit 1; }
H="$W/host"
J="$OUT/host.jsonl"
run() { nice -n 19 taskset -c "$1" env OMP_NUM_THREADS="$2" OMP_PROC_BIND=close "${@:3}" >>"$J" 2>>"$OUT/host.err"; }
run 16 1 "$H" stream 1
run 16-31 16 "$H" stream 16
run 16-23,144-151 16 "$H" stream 16 .two_sockets
run 16 1 "$H" amx 1 3
run 16-31 16 "$H" amx 16 3
for base in /tmp /workspace/tmp; do
  [ -d "$base" ] || continue
  d=$(mktemp -d -p "$base")
  run 16 1 "$H" disk "$d/seq.bin" 4096
  rm -rf "$d"
done
echo "host section done"
