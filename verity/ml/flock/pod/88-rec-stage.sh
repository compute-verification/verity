#!/usr/bin/env bash
# Staging UniversalUnit_v1{G, N_IN, N_OUT} as the recursion's inner statement at a real private circuit's size (rec_stage.py).
#   MODE=measure  today's path (rec_inner.universal's staging) on each class of LADDER, side by side, one process each:
#                 $OUT/measure-G-N_IN-N_OUT.jsonl (each phase's seconds and peak RSS) and .log; PROFILE_G: that class
#                 under cProfile as well ($OUT/profile-*.txt)
# Run from the verity tree (research run --cwd source). Big files under $FLOCK_WORK (default /workspace/jobs/rec-stage).
# env: LADDER="128,64,32 256,64,32 512,64,32 1024,64,32 2048,64,32"  PROFILE_G=128  N=8  CPUS=16  TIMEOUT=10800 (seconds, per class)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RS_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-16}" || exit 1
  export RS_PINNED=$SLICE_CPUS
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/rec-stage-job}/out}; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/jobs/rec-stage}; mkdir -p $FW || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
UV=$(command -v uv || echo $HOME/.local/bin/uv)
PYV=$FW/py
[ -x $PYV/bin/python ] || { $UV venv -q --python 3.12 $PYV && $UV pip install -q --python $PYV/bin/python numpy blake3; } || exit 1
export PATH=$PYV/bin:$PATH PYTHONPATH=$REPO:$REPO/catalog:$REPO/kernels:$REPO/verity/ml/flock/python
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$RS_PINNED"; free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
trap 'rm -rf $S' EXIT
PY=verity/ml/flock/pod/rec_stage.py
MODE=${MODE:-measure}; N=${N:-8}; TIMEOUT=${TIMEOUT:-10800}

if [ $MODE = measure ]; then
  pids=()
  for c in ${LADDER:-128,64,32 256,64,32 512,64,32 1024,64,32 2048,64,32}; do
    t=${c//,/-}
    prof=(); [ "${PROFILE_G:-128}" = "${c%%,*}" ] && prof=(--profile $S/profile-$t.prof)
    ( timeout $TIMEOUT /usr/bin/time -f '%e s %M KiB' -o $O/measure-$t.time python3 $PY measure --cls $c --n $N --out $S/$t "${prof[@]}" \
        > $O/measure-$t.jsonl 2> $O/measure-$t.log
      echo "exit=$?" >> $O/measure-$t.time
      ls -l $S/$t >> $O/measure-$t.log 2>&1
      [ -n "${prof[*]}" ] && python3 -c "import pstats, sys; pstats.Stats(sys.argv[1], stream=open(sys.argv[2], 'w')).sort_stats(sys.argv[3]).print_stats(40)" \
        $S/profile-$t.prof $O/profile-$t-tottime.txt tottime
      [ -n "${prof[*]}" ] && python3 -c "import pstats, sys; pstats.Stats(sys.argv[1], stream=open(sys.argv[2], 'w')).sort_stats(sys.argv[3]).print_stats(60)" \
        $S/profile-$t.prof $O/profile-$t-cumtime.txt cumtime
      rm -rf $S/$t ) &
    pids+=($!)
  done
  wait "${pids[@]}"
  grep -h RESULT $O/measure-*.jsonl | sed 's/^RESULT //' > $O/summary.jsonl
  exit 0
fi
echo "unknown MODE=$MODE" >&2; exit 2
