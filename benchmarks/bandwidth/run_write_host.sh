#!/usr/bin/env bash
# The write ceilings' host half (cwd = the source tree): builds write_host.c (bwwrite_host) and, on the CPUs the job was
# given, writes host DRAM with KERNELS (default nt st stosb) at one thread a core, for 1, 2, 4, ... cores and then every
# core given, by NUMA node and together, REPS times (default 2) for SECONDS each (default 2), MIB a thread (default 512,
# above a socket's L3 at one thread). Outputs in $OUT: env.txt, build.txt, write_host.jsonl, rc.txt.
#
#   research run --on auto --cpus 48 --mem-gb 64 --project verity --cwd source \
#     -- bash benchmarks/bandwidth/run_write_host.sh
set -uxo pipefail
O=${OUT:?run under research run}; mkdir -p "$O"
BW=$PWD/benchmarks/bandwidth
B=$(mktemp -d /tmp/bwwrite-host.XXXXXX); trap 'rm -rf "$B"' EXIT
{ date -u +%FT%TZ; lscpu; numactl -H; taskset -pc $$; grep -H . /proc/self/cgroup; head -20 /proc/meminfo
  cat /sys/kernel/mm/transparent_hugepage/enabled; sudo -n dmidecode -t memory | grep -E "Speed|Type:|Size" | sort | uniq -c
  uptime; } > "$O/env.txt" 2>&1
if ! gcc -O3 -march=native -pthread -Wall -o "$B/bwwrite_host" "$BW/write_host.c" > "$O/build.txt" 2>&1; then
  cat "$O/build.txt"; echo 4 > "$O/rc.txt"; exit 4
fi
sha256sum "$B/bwwrite_host" "$BW/write_host.c" >> "$O/build.txt"

# One CPU a physical core among those allowed, as "node cpu" lines in core order.
python3 - > "$B/cores.txt" <<'EOF'
import os
seen = set()
for c in sorted(os.sched_getaffinity(0)):
    t = f"/sys/devices/system/cpu/cpu{c}/topology"
    key = (open(f"{t}/physical_package_id").read().strip(), open(f"{t}/core_id").read().strip())
    if key in seen:
        continue
    seen.add(key)
    node = next(int(d[4:]) for d in os.listdir(f"/sys/devices/system/cpu/cpu{c}") if d.startswith("node"))
    print(node, c)
EOF
cat "$B/cores.txt" >> "$O/env.txt"
first() { head -n "$1" | awk '{print $2}' | paste -sd,; }
ALL=$(first 100000 < "$B/cores.txt")
N=$(wc -l < "$B/cores.txt")
SETS=()
for k in 1 2 4 8 16 32 64; do [ "$k" -lt "$N" ] && SETS+=("cores$k $(first "$k" < "$B/cores.txt")"); done
for nd in $(awk '{print $1}' "$B/cores.txt" | sort -u); do
  SETS+=("node$nd-all $(awk -v n="$nd" '$1 == n {print $2}' "$B/cores.txt" | paste -sd,)")
done
[ "$(awk '{print $1}' "$B/cores.txt" | sort -u | wc -l)" -gt 1 ] && SETS+=("all $ALL")
rc=0
for r in $(seq "${REPS:-2}"); do
  for k in ${KERNELS:-nt st stosb}; do
    for s in "${SETS[@]}"; do
      "$B/bwwrite_host" --label "${s%% *}" --cpus "${s#* }" --kernel "$k" --mib "${MIB:-512}" --seconds "${SECONDS_EACH:-2}" \
        | tee -a "$O/write_host.jsonl"
      [ "${PIPESTATUS[0]}" = 0 ] || rc=6
    done
  done
done
uptime >> "$O/env.txt"
echo "$rc" > "$O/rc.txt"
exit "$rc"
