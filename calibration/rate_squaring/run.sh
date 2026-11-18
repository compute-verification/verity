#!/usr/bin/env bash
# `verity calibrate rate_squaring sq|ifma`: this directory's one runner (cwd = the source tree), the command of a
# `research run`. Each writes its records in $RESEARCH_RUN_DIR/out and the run's result.json through the harness's kit
# (`result.py`), mod 2^$BITS - $C (default 2^512 - 569):
#
#   sq      sq.cu on one GPU, with toolkit 13.0 ($CUDA_HOME, else the machine's record: `machine.py toolkit 13.0`;
#           refused without one), built for ARCH (90a, 120a; where it isn't given, the node's first card's name tells
#           it), for every (FOLD, ILP) in $FORMS, each gated against Python's pow (512 words at K = 1, 777 and 1,000)
#           before it is timed, with the SASS opcode mix of every kernel, ncu's pipe and issue counts and its stall
#           reasons; then each pipe probe in $PROBES (P:W:A; set it empty for none), its digests checked against the
#           host's before it is timed. Timed and profiled in blocks of $TPB threads (default 128, the most the kernels
#           take); the gates use 128. Its result is the gates' validation, with no number.
#   ifma    ifma.c on the host CPUs (CPUs from the affinity mask): for G = 1, 2 and 3 vectors a thread, each compiler's
#           build (gcc, and clang where the node has it) gated against Python's pow (384 words at K = 1, 777 and 1,000)
#           before it is timed: on one thread, on one thread a physical core of the mask, and on every CPU of the mask.
#           Its result is the fastest timing as metrics.toml's number where every build passed its gate (else exit 5).
#
#   GPU_LEASE_WHO=ra-squaring-ceiling taskset -c 48-95 verity calibrate rate_squaring sq
#   taskset -c 48-95 verity calibrate rate_squaring ifma
set -uo pipefail
USAGE="usage: verity calibrate rate_squaring sq|ifma"

sq() {
set -x
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
CUDA_HOME=$(python3 benchmarks/kernels/machine.py toolkit 13.0) || exit 3
export PATH=$CUDA_HOME/bin:$PATH
export D=calibration/rate_squaring
export B; B=$(mktemp -d /tmp/rate-squaring.XXXXXX); trap 'rm -rf "$B"' EXIT
export FORMS=${FORMS:-"0:1 0:2 1:1 1:2 1:4 2:1 2:2 2:4 3:1 3:2 4:1 4:2"}
export PROBES=${PROBES-"0:16:0 0:16:16 0:16:32 1:16:0 1:16:16"}
BITS=${BITS:-512}; C=${C:-569}; export TPB=${TPB:-128}
if [ -z "${ARCH:-}" ]; then
  case $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2> /dev/null || true) in
    *H200* | *H100*) ARCH=90a ;;
    *"RTX PRO 6000"*) ARCH=120a ;;
    *) echo "sq: ARCH (90a, 120a) is not given and the card's name does not tell it" >&2; exit 3 ;;
  esac
fi
nvcc -O3 -gencode "arch=compute_$ARCH,code=sm_$ARCH" -std=c++17 -Xptxas -v -DSQ_BITS="$BITS" -DSQ_C="$C" -o "$B/sq" $D/sq.cu \
  > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
cuobjdump -sass "$B/sq" > "$B/sq.sass" 2>&1
python3 $D/sass_mix.py "$B/sq.sass" > "$O/sass-mix.json"
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/sq" clock > "$O/clock-before.json" 2>&1
  for fi in $FORMS; do
    f=${fi%%:*}; i=${fi##*:}; ok=1
    for k in 1 777 1000; do
      "$B/sq" gate $f $i $k 512 > "$B/gate-$f-$i-$k.txt" 2>> "$O/errors.txt" || echo "gate $f $i $k rc $?" >> "$O/errors.txt"
      python3 $D/sq_gate.py < "$B/gate-$f-$i-$k.txt" > "$O/gate-f$f-i$i-k$k.json" 2>&1 || ok=0
    done
    if [ $ok = 1 ]; then
      for w in 2 4; do
        "$B/sq" time $f $i 4000 $TPB $w 5 >> "$O/time.jsonl" 2>> "$O/errors.txt" || echo "time $f $i $w rc $?" >> "$O/errors.txt"
      done
    else
      echo "fold $f ilp $i failed its gate: not timed" >> "$O/errors.txt"
    fi
  done
  for pwa in $PROBES; do
    IFS=: read -r p w a <<< "$pwa"
    "$B/sq" probe $p $w $a 20000 4 >> "$O/probe.jsonl" 2>> "$O/errors.txt" || echo "probe $pwa rc $?" >> "$O/errors.txt"
  done
  "$B/sq" clock > "$O/clock-after.json" 2>&1
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  M=smsp__inst_executed.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.per_cycle_active,sm__cycles_elapsed.avg
  for fi in $FORMS; do
    f=${fi%%:*}; i=${fi##*:}
    ncu --metrics $M --csv -k regex:chain -c 1 "$B/sq" time $f $i 400 $TPB 2 1 > "$O/ncu-f$f-i$i.csv" 2>&1 || echo "ncu $f $i rc $?" >> "$O/errors.txt"
    ncu --section WarpStateStats --csv --page raw -k regex:chain -c 1 "$B/sq" time $f $i 400 $TPB 2 1 \
      > "$O/ncu-stalls-f$f-i$i.csv" 2>&1 || echo "ncu stalls $f $i rc $?" >> "$O/errors.txt"
  done
  for pwa in $PROBES; do
    IFS=: read -r p w a <<< "$pwa"
    ncu --metrics $M --csv -k regex:probe --launch-skip 1 -c 1 "$B/sq" probe $p $w $a 2000 4 > "$O/ncu-probe-$p-$w-$a.csv" 2>&1 || echo "ncu probe $pwa rc $?" >> "$O/errors.txt"
  done'
  rc=$?
  uv run -q python $D/result.py sq "$O"
  r=$?
  [ "$rc" = 0 ] || exit "$rc"
  exit "$r"
}

ifma() {
set -x
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
D=calibration/rate_squaring
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
uv run -q python $D/result.py ifma "$O"
}

case "${1:-}:$#" in
  sq:1 | ifma:1) "$1" ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
