#!/usr/bin/env bash
# The adversary's best host DRAM read rate on the CPUs this run is given (no GPU), as one Attempt (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/host_tier/run_dram.sh
#
# Sweeps dram.c's read kernels and huge pages at every CPU, the thread count (one a core, then SMT siblings), and the
# placement (local, the other socket's DRAM over UPI, interleaved, and local plus remote at once from two processes). Each
# configuration runs REPS times for SECONDS; each line records how busy the node's other CPUs were meanwhile.
# Env: SECONDS_EACH (3), REPS (3), MIB (per thread, 256), CPUS (the affinity's list unless given).
# Outputs under $RESEARCH_RUN_DIR/out: env.txt, dram.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
B=$(mktemp -d /tmp/host-tier-dram.XXXXXX); trap 'rm -rf "$B"' EXIT
S=${SECONDS_EACH:-3} R=${REPS:-3} M=${MIB:-256}
CPUS=${CPUS:-$(taskset -pc $$ | sed 's/.*: //')}
{ lscpu; numactl -H; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag;
  uptime; echo "cpus: $CPUS"; } > "$O/env.txt" 2>&1
gcc -O3 -march=native -pthread -Wall -o "$B/dram" dram.c || exit 4

# CPUs of the affinity: all of them, one thread a physical core (the lowest sibling), and the first n of each list.
ALL=$(python3 -c "
import sys
s=set()
for p in '$CPUS'.split(','):
    lo,_,hi=p.partition('-'); s.update(range(int(lo),int(hi or lo)+1))
print(','.join(map(str,sorted(s))))")
PHYS=$(python3 -c "
cpus=[int(c) for c in '$ALL'.split(',')]
def sib(c): return min(int(x) for p in open(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list').read().strip().split(',') for x in range(int(p.split('-')[0]), int(p.split('-')[-1])+1))
print(','.join(str(c) for c in cpus if sib(c)==c))")
first() { echo "$2" | tr , '\n' | head -n "$1" | paste -sd,; }
echo "all: $ALL" >> "$O/env.txt"; echo "phys: $PHYS" >> "$O/env.txt"

one() {  # label cpus node kernel thp
  python3 cpuload.py snap > "$B/a.json"
  line=$("$B/dram" --label "$1" --cpus "$2" --node "$3" --kernel "$4" --thp "$5" --mib "$M" --seconds "$S")
  load=$(python3 cpuload.py diff "$B/a.json" --exclude "$ALL")
  echo "${line%\}}, \"load\": $load}" | tee -a "$O/dram.jsonl"
}

nall=$(echo "$ALL" | tr , '\n' | wc -l); nphys=$(echo "$PHYS" | tr , '\n' | wc -l)
for r in $(seq "$R"); do
  for k in scalar avx512 nt pf pfnta multi; do for t in 0 1; do one kernels "$ALL" 0 $k $t; done; done
done
for r in $(seq "$R"); do
  for n in 1 2 4 8 12 16 $nphys; do
    [ "$n" -le "$nphys" ] && { one threads-phys "$(first $n "$PHYS")" 0 avx512 1; one threads-phys "$(first $n "$PHYS")" 0 pf 1; }
  done
  for n in 24 36 $nall; do
    [ "$n" -le "$nall" ] && { one threads-smt "$(first $n "$ALL")" 0 avx512 1; one threads-smt "$(first $n "$ALL")" 0 pf 1; }
  done
done
for r in $(seq "$R"); do
  one place "$ALL" 1 avx512 1
  one place "$PHYS" 1 avx512 1
  one place "$ALL" i avx512 1
  # local and remote at once: the two halves of the affinity, one process each
  h=$((nall / 2)); A=$(first $h "$ALL"); Bc=$(echo "$ALL" | tr , '\n' | tail -n +$((h + 1)) | paste -sd,)
  python3 cpuload.py snap > "$B/a.json"; at=$(python3 -c 'import time; print(time.time() + 4)')
  "$B/dram" --label both-local --cpus "$A" --node 0 --kernel avx512 --thp 1 --mib "$M" --seconds "$S" --start-at "$at" > "$B/l.json" &
  "$B/dram" --label both-remote --cpus "$Bc" --node 1 --kernel avx512 --thp 1 --mib "$M" --seconds "$S" --start-at "$at" > "$B/r.json" &
  wait
  load=$(python3 cpuload.py diff "$B/a.json" --exclude "$ALL")
  for f in l r; do l=$(cat "$B/$f.json"); echo "${l%\}}, \"load\": $load}" | tee -a "$O/dram.jsonl"; done
done
