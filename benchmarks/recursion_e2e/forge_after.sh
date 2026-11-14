#!/usr/bin/env bash
# forge-after (forge_after.py) over the releases of a run behind the ZK server, with that run's Lean verifier (its
# out/lean.sha) and this tree's (main's Lean sources), built under WORK; nothing under FLOCK_WORK is written. On the node that ran it, through a slot:
#   research run --on NODE --project verity --source . --cwd source -- env CHECK_SLOTS=/workspace/research/locks/leanrec/slots \
#     python3 tools/verity/check/slot.py --by benchmarks -- bash benchmarks/recursion_e2e/forge_after.sh \
#     FLOCK_WORK=/workspace/cache/bench-recrun-fw TAG=k64n16fw RUN=<the run id>
for kv in "$@"; do export "$kv"; done
: ${FLOCK_WORK:?} ${TAG:?} ${RUN:?}
OUT=${RESEARCH_RUN_DIR:-/tmp/forge-after}/out; export OUT; mkdir -p $OUT
WORK=${WORK:-/workspace/cache/forge-after}; mkdir -p $WORK
RUN_OUT=/workspace/research/runs/$RUN/out
want=$(cut -d' ' -f1 $RUN_OUT/lean.sha)
OLD=$(for b in $FLOCK_WORK/cache/lean-*/verity-verify-proof; do [ "$(sha256sum $b | cut -d' ' -f1)" = "$want" ] && echo $b; done | head -1)
[ -n "$OLD" ] || { echo "no Lean verifier under $FLOCK_WORK/cache/lean-* with $RUN's lean.sha $want"; exit 1; }
LSHA=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py key verity/protocols/proofs/flock/verifier)
NEW=$WORK/lean-$LSHA/verity-verify-proof
export PATH=$HOME/.elan/bin:$PATH
if [ ! -x $NEW ]; then
  ( exec 9>$WORK/lean.lock; flock 9
    [ -x $NEW ] || { rm -rf $WORK/lean-src && LEAN_SRC=$(python3 verity/protocols/proofs/flock/prover/pod/lake_closure.py copy verity/protocols/proofs/flock/verifier $WORK/lean-src) \
      && (cd $LEAN_SRC && lake build verity-verify-proof) && mkdir -p $WORK/lean-$LSHA && cp $LEAN_SRC/.lake/build/bin/verity-verify-proof $NEW; }
  ) > $OUT/build-lean.log 2>&1 || { tail -20 $OUT/build-lean.log; exit 1; }
fi
echo "run's Lean verifier $OLD; this tree's $NEW (lean-$LSHA)"
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)/tools/config:$(pwd)
python3 benchmarks/recursion_e2e/forge_after.py --work $FLOCK_WORK --tag $TAG --run-out $RUN_OUT --bin run=$OLD --bin main=$NEW
