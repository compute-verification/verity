#!/usr/bin/env bash
# forge-after (forge_after.py) over a firewalled run's releases, with that run's flock-verify (its out/lean.sha) and this
# tree's (main's Lean sources), built under WORK; nothing under FLOCK_WORK is written. On the node that ran it, through a slot:
#   research run --on NODE --project verity --source . --cwd source -- env CHECK_SLOTS=/workspace/research/locks/leanrec/slots \
#     python3 tools/verity/check/slot.py --by benchmarks -- bash benchmarks/recursion_e2e/forge_after.sh \
#     FLOCK_WORK=/workspace/cache/bench-recrun-fw TAG=k64n16fw RUN=<the run id>
for kv in "$@"; do export "$kv"; done
: ${FLOCK_WORK:?} ${TAG:?} ${RUN:?}
OUT=${RESEARCH_RUN_DIR:-/tmp/forge-after}/out; export OUT; mkdir -p $OUT
WORK=${WORK:-/workspace/cache/forge-after}; mkdir -p $WORK
RUN_OUT=/workspace/research/runs/$RUN/out
want=$(cut -d' ' -f1 $RUN_OUT/lean.sha)
OLD=$(for b in $FLOCK_WORK/cache/lean-*/flock-verify; do [ "$(sha256sum $b | cut -d' ' -f1)" = "$want" ] && echo $b; done | head -1)
[ -n "$OLD" ] || { echo "no flock-verify under $FLOCK_WORK/cache with $RUN's lean.sha $want"; exit 1; }
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
NEW=$WORK/lean-$LSHA/flock-verify
export PATH=$HOME/.elan/bin:$PATH
if [ ! -x $NEW ]; then
  ( exec 9>$WORK/lean.lock; flock 9
    [ -x $NEW ] || { rm -rf $WORK/lean-src && mkdir -p $WORK/lean-src && cp -r verity/core/service/flock/. $WORK/lean-src/ \
      && (cd $WORK/lean-src && lake build flock-verify) && mkdir -p $WORK/lean-$LSHA && cp $WORK/lean-src/.lake/build/bin/flock-verify $NEW; }
  ) > $OUT/build-lean.log 2>&1 || { tail -20 $OUT/build-lean.log; exit 1; }
fi
echo "run's flock-verify $OLD; this tree's $NEW (lean-$LSHA)"
python3 benchmarks/recursion_e2e/forge_after.py --work $FLOCK_WORK --tag $TAG --run-out $RUN_OUT --bin run=$OLD --bin main=$NEW
