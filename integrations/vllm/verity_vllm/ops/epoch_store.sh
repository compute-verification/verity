#!/bin/bash
# epoch_store.sh ROWDIR: a row's Build and records as fixture/v1 trees, `research data put --preserve` with this run's own custody key
# (research run --custody-r2 stages it under requests/<run id>/custody/; the pod holds no other credential), then
# `research data preserved` on every tree it stored.  Run by epoch_row.sh inside the row's job, on every end of the row.
#   build    the row dir without match/, commit/ and vu-export/: Programs, workload Program, manifest, summaries, logs (files <= 8 GB)
#   records  match/ and commit/ files under 200 MB, and the row's verdict / stage / log files
#   capture  match/capture/ as one tar, when it holds more than CAPTURE_TAR_FILES files (default 100000): a TP row's capture is ~0.5M
#            small files, which a per-file upload does not finish inside a pod's life; the records tree then leaves it out
#   large    (STORE_KINDS=... large) the match/ and commit/ files over 200 MB
# Staged as hardlinks under $RESEARCH_STORE/../stage/<run id>/; what is left out is listed in evidence/store_omitted.txt.
# Prints `STORED <kind> art:<id> <PRESERVED|FAILED>` per tree and `PRESERVED-CHECK <ok|FAILED> <arts>`; exit 0 only when every tree
# asked for is PRESERVED and the check agrees.
set -u
D=${1:?row dir}; ROWID=$(basename "$D")
EV=$RESEARCH_RUN_DIR/evidence
R=${RESEARCH_CLI:-python3 -m research}
VV=${VV:-${PY:-/workspace/venv312/bin/python}}; [[ "$VV" == */verity-vllm ]] || VV=${VV%/*}/verity-vllm
C=${CUSTODY_DIR:-$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/${RESEARCH_RUN_ID:?}/custody}
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$("$VV" epoch custody-env "$C")"
S=${RESEARCH_STORE:-/workspace/epoch/store}/../stage/$RESEARCH_RUN_ID; rm -rf "$S"; mkdir -p "$S/build" "$S/records" "$S/large"

cp -al "$D/." "$S/build/"
rm -rf "$S/build/match" "$S/build/commit" "$S/build/vu-export"
: > "$EV/store_omitted.txt"
find "$S/build" -type f -size +8G -printf "build %s %P\n" -delete >> "$EV/store_omitted.txt"
(cd "$D" && find match commit -type f -size -200M -print0 2>/dev/null | xargs -0 -r cp -al --parents -t "$S/records/")
(cd "$D" && find match commit -type f -size +200M -printf "records %s %p\n" 2>/dev/null >> "$EV/store_omitted.txt")
(cd "$D" && find match commit -type f -size +200M -print0 2>/dev/null | xargs -0 -r cp -al --parents -t "$S/large/")
for f in verdict.json stages.txt row.log timeline.jsonl commit.log build_summary.json target_family.json admission.json; do
  [ -f "$D/$f" ] && cp -al "$D/$f" "$S/records/"
done
KINDS=${STORE_KINDS:-build records}
if [ -d "$S/records/match/capture" ] && [ "$(find "$S/records/match/capture" -type f | wc -l)" -gt "${CAPTURE_TAR_FILES:-100000}" ]; then
  rm -rf "$S/records/match/capture"; mkdir -p "$S/capture"; tar cf "$S/capture/match-capture.tar" -C "$D" match/capture
  [[ " $KINDS " == *" capture "* ]] || KINDS="$KINDS capture"
fi

meta() {  # meta KIND ROLE -> the tree's meta (serving-view's record-build layout), from the Build's own summaries
  "$VV" epoch store-meta "$D" --kind "$1" --role "$2" --rownum "$ROWNUM" --run-id "$RESEARCH_RUN_ID" --epoch-sha "$EPOCH_SHA" \
    --gpu "${GPU_NAME:-}" --pod "${POD:-}" --epoch "${EPOCH_NAME:-vllm-rebaseline-epoch}"
}

rc=0; arts=""
for kind in $KINDS; do
  case "$kind" in records) role=records;; large) role=records-large;; capture) role=records-capture;; *) role=programs;; esac
  meta "$kind" "$role" > "$EV/meta-$kind.json"
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
rm -rf "$S"
exit $rc
