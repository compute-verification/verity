#!/usr/bin/env bash
# The door's walls (`door_walls.py`) on a CPU node: Lean's flock-lower-check built once per source of Flock's verifier
# (cached under its `lake_closure.py` key), then the census on each CIRCUITS item (NAME:BLOCKS[:UNITS[:FORM]]), each CASES
# name (`door_walls.py case`), and the fuzzer for FUZZ_MINUTES from FUZZ_SEED under FUZZ_FORM; NO_ROWS=1 skips rows. Each census's table, each
# case and the fuzzer's summary go to $OUT.
# research run --on auto --cwd clone --mem-gb 32 --cpus 8 --timeout 2h -- bash benchmarks/one_stage/door_walls.sh
set -uo pipefail
CIRCUITS=${CIRCUITS:-"one_linear:one-per-template test_class:shapes"}
FUZZ_MINUTES=${FUZZ_MINUTES:-0}
FUZZ_SEED=${FUZZ_SEED:-1}
FUZZ_CASES=${FUZZ_CASES:-100000}
FUZZ_FORM=${FUZZ_FORM:-unit}
CASES=${CASES:-}
O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/door-walls}/out}; mkdir -p $O
UV=$(command -v uv || echo $HOME/.local/bin/uv)
export PATH=$HOME/.elan/bin:$PATH
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
L=${DOOR_WALLS_CACHE:-$HOME/.cache/door-walls}; mkdir -p $L
LSHA=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py key verity/protocols/proof/flock/verifier)
LOW=$L/lean-$LSHA/flock-lower-check
(
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LOW ]; then
    S=$(mktemp -d); trap 'rm -rf $S' EXIT
    SRC=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py copy verity/protocols/proof/flock/verifier $S) \
      && (cd $SRC && lake build flock-lower-check) && mkdir -p $L/lean-$LSHA && cp $SRC/.lake/build/bin/flock-lower-check $LOW
  fi
) > $O/build-lean.log 2>&1 || { tail -30 $O/build-lean.log; exit 1; }
echo "LOWER $LOW $(sha256sum $LOW | cut -c1-16)"
PIECE_ARG=""
if [ "${PIECE:-0}" = 1 ]; then
  SPOT=verity/protocols/proof/spot/verifier/lean
  PSHA=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py key $SPOT)
  PEX=$L/spot-$PSHA/spot-difftest
  (
    exec 9>$L/spot.lock; flock 9
    if [ ! -x $PEX ]; then
      S=$(mktemp -d); trap 'rm -rf $S' EXIT
      SRC=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py copy $SPOT $S) \
        && (cd $SRC && lake build spot-difftest) && mkdir -p $L/spot-$PSHA && cp $SRC/.lake/build/bin/spot-difftest $PEX
    fi
  ) > $O/build-spot.log 2>&1 || { tail -30 $O/build-spot.log; exit 1; }
  echo "PIECE $PEX $(sha256sum $PEX | cut -c1-16)"; PIECE_ARG="--piece $PEX"
fi
$UV sync -q --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 1; }
PY=.venv/bin/python
rc=0
for item in $CIRCUITS; do
  IFS=: read -r name blocks units form <<< "$item"; units=${units:-one}; form=${form:-unit}; f=$O/census-$name-$units-$form
  $PY benchmarks/one_stage/door_walls.py census --circuit $name --blocks $blocks --units $units --form $form $([ $form = whole ] && echo $PIECE_ARG) ${ONLY:+--only "$ONLY"} ${NO_ROWS:+--no-rows} --one-circuit "${ONE_CIRCUIT:-}" --lower $LOW --json $f.json \
    > $f.log 2>&1
  r=$?; echo "CENSUS $name $blocks $units $form rc $r"; grep -h '^BUILT\|^PLAN\|^CENSUS\|^|' $f.log | cut -c1-600
  [ $r -gt 1 ] && { tail -20 $f.log; rc=1; }
done
for name in $CASES; do
  $PY benchmarks/one_stage/door_walls.py case $name --lower $LOW > $O/case-$name.log 2>&1
  r=$?; echo "CASE $name rc $r"; grep -h '^CASE' $O/case-$name.log | cut -c1-3000
  [ $r -ne 0 ] && { tail -20 $O/case-$name.log; rc=1; }
done
if [ "$FUZZ_MINUTES" != 0 ]; then
  $PY benchmarks/one_stage/door_walls.py fuzz --seed $FUZZ_SEED --cases $FUZZ_CASES --minutes $FUZZ_MINUTES --form $FUZZ_FORM --lower $LOW \
    --json $O/fuzz.json > $O/fuzz.log 2>&1
  r=$?; echo "FUZZ rc $r"; grep -h '^SUMMARY\|^CLASS' $O/fuzz.log | cut -c1-1500
  [ $r -gt 1 ] && { tail -20 $O/fuzz.log; rc=1; }
fi
exit $rc
