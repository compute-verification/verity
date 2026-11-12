#!/bin/bash
# store_trees.sh STAGE KIND...: each tree STAGE/KIND as a fixture/v1 artifact under the meta $EV/meta-KIND.json, `research data put
# --preserve` with this run's own custody key (research run --custody-r2 stages it under requests/<run id>/custody/; the job holds no
# other credential), then `research data preserved` on every tree it stored. With LABEL_KEY set, the run is labelled LABEL_KEY = each
# stored tree's art id (`research data label --by LABEL_BY`, the caller's asserter, required). Run by epoch_store.sh and one_stage's
# a5_tp.sh.
# Prints `STORED <kind> art:<id> <PRESERVED|FAILED>` per tree and `PRESERVED-CHECK <ok|FAILED> <arts>`; exit 0 only when every tree
# asked for is PRESERVED, the check agrees and every label is written; exit 4 when the run has no custody key.
set -u
S=${1:?stage dir}; shift
EV=${EV:?the evidence dir: the meta-KIND.json of each tree, and the put records}
VV=${VV:?the verity-vllm CLI}
[ -z "${LABEL_KEY:-}" ] || : "${LABEL_BY:?with LABEL_KEY: the asserter of the labels (research data label --by)}"
R=${RESEARCH_CLI:-python3 -m research}
C=${CUSTODY_DIR:-$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/${RESEARCH_RUN_ID:?}/custody}
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$("$VV" epoch custody-env "$C")"

rc=0; arts=""
for kind in "$@"; do
  echo "=== $(date -u +%H:%M:%SZ) put $kind $(du -sh "$S/$kind" | cut -f1) $(find "$S/$kind" -type f | wc -l) files"
  ok=0
  for i in 1 2 3; do
    $R data put --kind fixture/v1 --tree "$S/$kind" --meta @"$EV/meta-$kind.json" --preserve --json \
      > "$EV/put-$kind.json" 2> "$EV/put-$kind.err" && { ok=1; break; }
    tail -n 3 "$EV/put-$kind.err"; sleep $((20 * i))
  done
  art=$(grep -o 'art:[0-9a-f]\{64\}' "$EV/put-$kind.json" | head -n 1)
  if [ "$ok" = 1 ] && [ -n "$art" ]; then echo "STORED $kind $art PRESERVED"; arts="$arts $art"; else echo "STORED $kind ${art:-none} FAILED"; rc=1; fi
done
if [ -n "$arts" ] && $R data preserved $arts > "$EV/preserved.log" 2>&1; then
  echo "PRESERVED-CHECK ok$arts"
else
  echo "PRESERVED-CHECK FAILED$arts (preserved.log)"; rc=1
fi
for art in $arts; do
  [ -z "${LABEL_KEY:-}" ] || $R data label "$RESEARCH_RUN_ID" "$LABEL_KEY" "$art" --by "$LABEL_BY" >> "$EV/label.log" 2>&1 \
    || { echo "LABEL FAILED $RESEARCH_RUN_ID $LABEL_KEY $art (label.log)"; rc=1; }
done
exit $rc
