#!/usr/bin/env bash
# Host section of census.sh: host.c's STREAM-like bandwidth, AMX tile throughput and disk sequential read, all under
# nice 19 with at most 16 threads, pinned to 16 physical cores of NUMA node 0 (the fill jobs use 96-127 on
# vy-nebius-2, NUMA node 1), plus lscpu,
# lsblk, NUMA layout and memory size. Usage: host.sh WORKDIR (a scratch directory census.sh deletes); writes
# $OUT/host.jsonl and $OUT/host-*.txt.
set -uo pipefail
OUT=${OUT:?}
W=${1:?workdir}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lscpu >"$OUT/host-lscpu.txt" 2>&1
lscpu -p=CPU,CORE,SOCKET,NODE >"$OUT/host-lscpu-p.txt" 2>&1
lsblk -o NAME,SIZE,TYPE,ROTA,MODEL,MOUNTPOINT >"$OUT/host-lsblk.txt" 2>&1
free -b >"$OUT/host-free.txt" 2>&1
(numactl -H || cat /sys/devices/system/node/node*/meminfo) >"$OUT/host-numa.txt" 2>&1
df -h /tmp /workspace >"$OUT/host-df.txt" 2>&1
gcc -O3 -march=native -fopenmp -mamx-tile -mamx-int8 -mamx-bf16 -o "$W/host" "$SRC/host.c" -lpthread >"$OUT/build-host.log" 2>&1 \
  || { echo "host build failed" >&2; exit 1; }
H="$W/host"
# 16 CPUs on distinct physical cores of NUMA node 0, from CPU 16 up, within this process's allowed set
allowed=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')
CPUS=$(python3 - "$allowed" <<'PY'
import subprocess, sys
ok = set()
for part in sys.argv[1].split(","):
    a, _, b = part.partition("-")
    ok.update(range(int(a), int(b or a) + 1))
seen, pick = set(), []
for line in subprocess.run(["lscpu", "-p=CPU,CORE,NODE"], capture_output=True, text=True).stdout.splitlines():
    if line.startswith("#"):
        continue
    cpu, core, node = (int(x) for x in line.split(","))
    if node == 0 and cpu >= 16 and cpu in ok and core not in seen:
        seen.add(core)
        pick.append(cpu)
print(",".join(map(str, pick[:16])))
PY
)
ONE=${CPUS%%,*}
echo "{\"host_cpus\":\"$CPUS\",\"allowed\":\"$allowed\"}" >"$OUT/host-cpus.json"
J="$OUT/host.jsonl"
run() { nice -n 19 taskset -c "$1" env OMP_NUM_THREADS="$2" OMP_PROC_BIND=close "${@:3}" >>"$J" 2>>"$OUT/host.err"; }
run "$ONE" 1 "$H" stream 1
run "$CPUS" 16 "$H" stream 16
run "$ONE" 1 "$H" amx 1 3
run "$CPUS" 16 "$H" amx 16 3
for base in /tmp /workspace/tmp; do
  [ -d "$base" ] || continue
  d=$(mktemp -d -p "$base")
  run "$ONE" 1 "$H" disk "$d/seq.bin" 4096 ".$(basename "$base")_$(df --output=source "$base" | tail -1 | xargs basename)"
  rm -rf "$d"
done
echo "host section done"
