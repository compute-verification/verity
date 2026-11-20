#!/usr/bin/env bash
# The door's walls (`door_walls.py`) on a CPU node: Lean's flock-lower-check built once per source of Flock's verifier
# (cached under its `lake_closure.py` key), then the census on each CIRCUITS item (NAME:BLOCKS) and the fuzzer for
# FUZZ_MINUTES from FUZZ_SEED. Each census's table and the fuzzer's summary go to $OUT.
# research run --on auto --cwd clone --mem-gb 32 --cpus 8 --timeout 2h -- bash benchmarks/one_stage/door_walls.sh
set -uo pipefail
CIRCUITS=${CIRCUITS:-"one_linear:one-per-template test_class:shapes"}
FUZZ_MINUTES=${FUZZ_MINUTES:-0}
FUZZ_SEED=${FUZZ_SEED:-1}
FUZZ_CASES=${FUZZ_CASES:-100000}
O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/door-walls}/out}; mkdir -p $O
UV=$(command -v uv || echo $HOME/.local/bin/uv)
export PATH=$HOME/.elan/bin:$PATH
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
L=${DOOR_WALLS_CACHE:-$HOME/.cache/door-walls}; mkdir -p $L
LSHA=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py key verity/protocols/proofs/flock/verifier)
LOW=$L/lean-$LSHA/flock-lower-check
(
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LOW ]; then
    S=$(mktemp -d); trap 'rm -rf $S' EXIT
    SRC=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py copy verity/protocols/proofs/flock/verifier $S) \
      && (cd $SRC && lake build flock-lower-check) && mkdir -p $L/lean-$LSHA && cp $SRC/.lake/build/bin/flock-lower-check $LOW
  fi
) > $O/build-lean.log 2>&1 || { tail -30 $O/build-lean.log; exit 1; }
echo "LOWER $LOW $(sha256sum $LOW | cut -c1-16)"
$UV sync -q --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 1; }
PY=.venv/bin/python
rc=0
for item in $CIRCUITS; do
  name=${item%%:*}; blocks=${item#*:}
  $PY benchmarks/one_stage/door_walls.py census --circuit $name --blocks $blocks --lower $LOW --json $O/census-$name.json \
    > $O/census-$name.log 2>&1
  r=$?; echo "CENSUS $name $blocks rc $r"; grep -h '^BUILT\|^CENSUS\|^|' $O/census-$name.log | cut -c1-600
  [ $r -gt 1 ] && { tail -20 $O/census-$name.log; rc=1; }
done
if [ "$FUZZ_MINUTES" != 0 ]; then
  $PY benchmarks/one_stage/door_walls.py fuzz --seed $FUZZ_SEED --cases $FUZZ_CASES --minutes $FUZZ_MINUTES --lower $LOW \
    --json $O/fuzz.json > $O/fuzz.log 2>&1
  r=$?; echo "FUZZ rc $r"; grep -h '^SUMMARY\|^CLASS' $O/fuzz.log | cut -c1-1500
  [ $r -gt 1 ] && { tail -20 $O/fuzz.log; rc=1; }
fi
exit $rc
