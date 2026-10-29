#!/usr/bin/env bash
# ifma.c on the host CPUs (cwd = the source tree, CPUs from the affinity mask), mod 2^$BITS - $C (default 2^512 - 569):
# for G = 1, 2 and 3 vectors a thread,
# each compiler's build (gcc, and clang where the node has it) gated against
# Python's pow (384 words at K = 1, 777 and 1,000) before it is timed: on one thread, on one thread a physical core of
# the mask, and on every CPU of the mask. Results in $RESEARCH_RUN_DIR/out.
#
#   RESEARCH_RUN_DIR=DIR taskset -c 48-95 bash benchmarks/rate_squaring/ifma.sh
set -uxo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
D=benchmarks/rate_squaring
B=$(mktemp -d /tmp/rate-squaring-ifma.XXXXXX); trap 'rm -rf "$B"' EXIT
BITS=${BITS:-512}; C=${C:-569}
{ lscpu; grep -m1 "model name" /proc/cpuinfo; grep -c processor /proc/cpuinfo; } > "$O/cpu.txt" 2>&1
MASK=$(taskset -pc $$ | sed 's/.*: //')
lscpu -e=CPU,CORE,SOCKET,NODE > "$O/topology.txt" 2>&1
# The first CPU of each physical core in the mask, and how many CPUs of the mask share a core.
CORES=$(python3 - "$MASK" <<'EOF'
import os, sys
mask = os.sched_getaffinity(0)
seen, first = set(), []
for c in sorted(mask):
    sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
    if sib not in seen:
        seen.add(sib)
        first.append(c)
print(",".join(map(str, first)))
EOF
)
echo "mask $MASK cores-first $CORES" >> "$O/topology.txt"
NCPU=$(nproc)
NCORE=$(echo "$CORES" | tr ',' '\n' | wc -l)
CCS=()
for cc in gcc clang; do command -v $cc > /dev/null && CCS+=("$cc"); done
for cc in "${CCS[@]}"; do
  name=$(basename "$cc")
  $cc --version 2>&1 | head -1 >> "$O/compilers.txt"
  for g in 1 2 3; do
    $cc -O3 -march=native -DG=$g -DIF_BITS="$BITS" -DIF_C="$C" -o "$B/ifma-$name-$g" $D/ifma.c -lpthread >> "$O/build.log" 2>&1 || { echo "build $name $g" >> "$O/errors.txt"; continue; }
    ok=1
    for k in 1 777 1000; do
      "$B/ifma-$name-$g" gate $k 384 | python3 $D/sq_gate.py > "$O/gate-$name-g$g-k$k.json" 2>&1 || ok=0
    done
    if [ $ok = 1 ]; then
      for r in 1 2 3; do "$B/ifma-$name-$g" time 3000000 1 | sed "s/^{/{\"cc\": \"$name\", \"set\": \"one\", /" >> "$O/time.jsonl"; done
      for r in 1 2; do
        taskset -c "$CORES" "$B/ifma-$name-$g" time 3000000 "$NCORE" | sed "s/^{/{\"cc\": \"$name\", \"set\": \"cores\", /" >> "$O/time.jsonl"
        "$B/ifma-$name-$g" time 3000000 "$NCPU" | sed "s/^{/{\"cc\": \"$name\", \"set\": \"mask\", /" >> "$O/time.jsonl"
      done
    else
      echo "$name g $g failed its gate: not timed" >> "$O/errors.txt"
    fi
  done
done
