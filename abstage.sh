#!/usr/bin/env bash
# RECFAST: V*'s staging A/B on recprof.sh's data. Each variant stages each statement of WHAT into a scratch dir under
# /usr/bin/time; then every staged file's sha256 per variant, and whether each variant's files equal the first's.
# VARIANTS="name=SRC[:gc-off] ..." (SRC a source tree; default: base (BASE_SRC) and this run's source, each also gc-off)
# env: FLOCK_WORK=/workspace/research/cache/recfast TAG=k64-n64 WHAT="alg L0" BASE_SRC=/workspace/research/src/<sha>
set -uo pipefail
SRC=$(pwd); OUT=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$OUT"
FW=${FLOCK_WORK:-/workspace/research/cache/recfast}; DD=$FW/data/${TAG:-k64-n64}; V=$DD/v; INNER=$DD/inner/s1
PY=$FW/flock-circuit/py/bin/python3
BASE=${BASE_SRC:?BASE_SRC=a source tree of the base}
VARIANTS=${VARIANTS:-"base=$BASE new=$SRC new-gcoff=$SRC:gc-off"}
T=$(mktemp -d $FW/abstage.XXXXXX); trap 'rm -rf "$T"' EXIT
run1() {  # name src gc what
  local name=$1 src=$2 gc=$3 w=$4 A
  case $w in alg) A=(--levels none) ;; L*) A=(--levels ${w#L} --no-algebra) ;; esac
  local pp=$src:$src/benchmarks/numerical/python:$src/verity/ml/flock/python:$src/integrations/vllm:$src/tools/verity/circuit_check/src
  local pre="import sys, runpy"; [ "$gc" = gc-off ] && pre="$pre, gc; gc.disable()"
  PYTHONPATH=$pp /usr/bin/time -v -o $OUT/time-$name-$w.txt $PY -c "$pre
sys.argv = ['rec_vstage'] + sys.argv[1:]
runpy.run_module('verity_flock.rec_vstage', run_name='__main__', alter_sys=True)" \
    stage $INNER $V/fold.json $T/$name/$w --points $V/points.json --salts $V/acc-salts.bin "${A[@]}" > $OUT/$name-$w.json 2> $OUT/$name-$w.err
  echo "$name $w exit $?"
}
export -f run1; export OUT T INNER V PY
for w in ${WHAT:-alg L0}; do
  for v in $VARIANTS; do
    name=${v%%=*}; rest=${v#*=}; src=${rest%%:*}; gc=; [[ $rest == *:gc-off ]] && gc=gc-off
    run1 $name $src "$gc" $w &
  done
  wait
done
for v in $VARIANTS; do
  name=${v%%=*}
  (cd $T/$name && find . -type f | LC_ALL=C sort | xargs sha256sum) > $OUT/sha-$name.txt
done
first=${VARIANTS%% *}; first=${first%%=*}
for v in $VARIANTS; do
  name=${v%%=*}
  if cmp -s $OUT/sha-$first.txt $OUT/sha-$name.txt; then echo "$name: every file equals $first's"; else echo "$name: differs from $first:"; diff $OUT/sha-$first.txt $OUT/sha-$name.txt | head -20; fi
done | tee $OUT/identity.txt
for f in $OUT/time-*.txt; do printf '%s wall %s peak_kb %s cpu_user %s\n' "$(basename $f .txt)" "$(grep -oP 'Elapsed \(wall clock\) time.*: \K.*' $f)" \
  "$(grep -oP 'Maximum resident set size \(kbytes\): \K.*' $f)" "$(grep -oP 'User time \(seconds\): \K.*' $f)"; done | tee $OUT/times.txt
