#!/usr/bin/env bash
# Every run from this branch is a development run: the form's switch on, ahead of its proof; not main's verifier.
# two_stage_small.sh from a tree whose verifier allows the two-stage form ahead of its proof (SecondStage.FORM and
# second_stage.FORM on), to find what else stops a two-stage run. Every record under $OUT is labelled: LABEL.txt holds
# the label, the tree's commit and both switches as the tree states them. An accept from this tree is evidence about
# the stops, never a result. The rehearsal's own flags pass through (SMALL_ARGS, SMALL_FIRST, SMALL_SECOND,
# SMALL_INLINE); exit: the rehearsal's. STOPS_SCRIPT names another script to run under the label (the toy's
# benchmarks/two_stage/audit.sh, whose round 1 commits its replay block's return). STOPS_FIREWALL=1 puts every
# session of `second` behind the ZK server (--firewall, the flock-firewall the rehearsal builds).
#   research run --on auto --cwd clone --cpus 16 --mem-gb 64 --max-min 120 -- bash benchmarks/one_stage/two_stage_stops.sh
set -uo pipefail
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
LABEL="the form's switch on, ahead of its proof; not main's verifier"
SCRIPT=${STOPS_SCRIPT:-benchmarks/one_stage/two_stage_small.sh}
if [ -n "${STOPS_FIREWALL:-}" ]; then
  export SMALL_SECOND="${SMALL_SECOND:-} --firewall $REPO/verity/protocols/flock/verifier/.lake/build/bin/flock-firewall"
fi
{
  echo "LABEL: $LABEL"
  echo "commit: $(git rev-parse HEAD)"
  grep -n '^def FORM ' verity/protocols/spot/verifier/lean/Spot/SecondStage.lean
  grep -n '^FORM = ' verity/protocols/spot/verifier/second_stage.py
  echo "script: $SCRIPT"
  echo "flags: SMALL_INLINE=${SMALL_INLINE:-} SMALL_ARGS=${SMALL_ARGS:-} SMALL_FIRST=${SMALL_FIRST:-} SMALL_SECOND=${SMALL_SECOND:-} TWO_CASES=${TWO_CASES:-} TWO_ARGS=${TWO_ARGS:-}"
} | tee "$O/LABEL.txt"
grep -q '^def FORM : Bool := true$' verity/protocols/spot/verifier/lean/Spot/SecondStage.lean \
  && grep -q '^FORM = True$' verity/protocols/spot/verifier/second_stage.py \
  || { echo "the form's switch is not on in both the Lean and its mirror: this is not the run it says it is" | tee -a "$O/LABEL.txt"; exit 9; }
rc=0
bash "$SCRIPT" || rc=$?
echo "LABEL: $LABEL ($SCRIPT rc=$rc)" | tee -a "$O/LABEL.txt"
exit $rc
