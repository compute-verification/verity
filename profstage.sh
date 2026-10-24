#!/usr/bin/env bash
# RECFAST: V*'s staging under cProfile on recprof.sh's data (the inner session, fold, points and salts it left under
# FLOCK_WORK/data/TAG), each named statement into a scratch dir; pstats by cumulative and own time into $OUT.
# env: FLOCK_WORK=/workspace/research/cache/recfast TAG=k64-n64 WHAT="alg L0"
set -uo pipefail
SRC=$(pwd); OUT=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$OUT"
FW=${FLOCK_WORK:-/workspace/research/cache/recfast}; DD=$FW/data/${TAG:-k64-n64}; V=$DD/v; INNER=$DD/inner/s1
export PATH=$FW/flock-circuit/py/bin:$PATH PYTHONPATH=$SRC:$SRC/benchmarks/numerical/python:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src
T=$(mktemp -d $FW/profstage.XXXXXX); trap 'rm -rf "$T"' EXIT
for w in ${WHAT:-alg L0}; do
  case $w in alg) A=(--levels none) ;; L*) A=(--levels ${w#L} --no-algebra) ;; esac
  /usr/bin/time -v -o $OUT/time-$w.txt python3 -m cProfile -o $T/$w.prof -m verity_flock.rec_vstage stage $INNER $V/fold.json $T/$w \
    --points $V/points.json --salts $V/acc-salts.bin "${A[@]}" > $OUT/$w.json 2> $OUT/$w.err
  echo "$w exit $?"
  python3 -c "
import pstats, sys
p = pstats.Stats(sys.argv[1])
p.sort_stats('cumulative').print_stats(60)
p.sort_stats('tottime').print_stats(40)
p.sort_stats('tottime').print_callers(15)
" $T/$w.prof > $OUT/pstats-$w.txt 2>&1
  cp $T/$w.prof $OUT/
done
