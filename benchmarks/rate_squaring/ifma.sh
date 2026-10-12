#!/usr/bin/env bash
# ifma.c on the host CPUs (cwd = the source tree, CPUs from the affinity mask): for G = 1, 2 and 3 vectors a thread,
# clang's and gcc's builds gated against Python's pow (384 words at K = 1, 777 and 1,000) before they are timed, on one
# thread and on every CPU of the mask. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/rate_squaring/ifma.sh
set -uxo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
D=benchmarks/rate_squaring
B=$(mktemp -d /tmp/rate-squaring-ifma.XXXXXX); trap 'rm -rf "$B"' EXIT
{ lscpu; grep -m1 "model name" /proc/cpuinfo; grep -c processor /proc/cpuinfo; } > "$O/cpu.txt" 2>&1
NCPU=$(nproc)
for cc in clang gcc; do
  command -v $cc > /dev/null || { echo "no $cc" >> "$O/errors.txt"; continue; }
  $cc --version | head -1 >> "$O/compilers.txt"
  for g in 1 2 3; do
    $cc -O3 -march=native -DG=$g -o "$B/ifma-$cc-$g" $D/ifma.c -lpthread >> "$O/build.log" 2>&1 || { echo "build $cc $g" >> "$O/errors.txt"; continue; }
    ok=1
    for k in 1 777 1000; do
      "$B/ifma-$cc-$g" gate $k 384 | python3 $D/sq_gate.py > "$O/gate-$cc-g$g-k$k.json" 2>&1 || ok=0
    done
    if [ $ok = 1 ]; then
      for r in 1 2 3; do "$B/ifma-$cc-$g" time 3000000 1 >> "$O/time.jsonl"; done
      for r in 1 2; do "$B/ifma-$cc-$g" time 3000000 "$NCPU" >> "$O/time.jsonl"; done
      grep MHz /proc/cpuinfo | sort | uniq -c | sort -rn | head -3 >> "$O/mhz-after-$cc-$g.txt"
    else
      echo "$cc g $g failed its gate: not timed" >> "$O/errors.txt"
    fi
  done
done
