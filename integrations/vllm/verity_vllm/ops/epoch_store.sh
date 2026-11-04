#!/bin/bash
# epoch_store.sh ROWDIR: a workload's Build and records as fixture/v1 trees, each under its meta (serving-view's record-build layout), stored
# by store_trees.sh (`research data put --preserve` with this run's own custody key, then `research data preserved` on every tree it
# stored).  Run by epoch_row.sh inside the workload's job, on every end of it.
#   build    the workload dir without match/, commit/ and vu-export/: Programs, workload Program, manifest, summaries, logs (files <= 8 GB)
#   records  match/ and commit/ files under 200 MB, and the workload's verdict / stage / log files
#   capture  match/capture/ as one tar, when it holds more than CAPTURE_TAR_FILES files (default 100000): a TP workload's capture is ~0.5M
#            small files, which a per-file upload does not finish inside a pod's life; the records tree then leaves it out
#   large    (STORE_KINDS=... large) the match/ and commit/ files over 200 MB
# No tree holds a replay bundle or a slim keep (clear values with their salts), nor where each rank ran (topology.json) or when each
# process ran (timeline.jsonl, timeline_report.json): those stay on the node, prover-side.
# Staged as hardlinks under $RESEARCH_STORE/../stage/<run id>/; what is left out is listed in evidence/store_omitted.txt.
# Prints store_trees.sh's `STORED <kind> art:<id> <PRESERVED|FAILED>` per tree and `PRESERVED-CHECK <ok|FAILED> <arts>`, and exits
# with its code: 0 only when every tree asked for is PRESERVED and the check agrees.
set -u
D=${1:?row dir}; ROWID=$(basename "$D")
export EV=$RESEARCH_RUN_DIR/evidence
VV=${VV:-${PY:-/workspace/venv312/bin/python}}; [[ "$VV" == */verity-vllm ]] || VV=${VV%/*}/verity-vllm; export VV
S=${RESEARCH_STORE:-/workspace/epoch/store}/../stage/$RESEARCH_RUN_ID; rm -rf "$S"; mkdir -p "$S/build" "$S/records" "$S/large"

cp -al "$D/." "$S/build/"
rm -rf "$S/build/match" "$S/build/commit" "$S/build/vu-export"
: > "$EV/store_omitted.txt"
find "$S/build" -type f -size +8G -printf "build %s %P\n" -delete >> "$EV/store_omitted.txt"
(cd "$D" && find match commit -type f -size -200M -print0 2>/dev/null | xargs -0 -r cp -al --parents -t "$S/records/")
(cd "$D" && find match commit -type f -size +200M -printf "records %s %p\n" 2>/dev/null >> "$EV/store_omitted.txt")
(cd "$D" && find match commit -type f -size +200M -print0 2>/dev/null | xargs -0 -r cp -al --parents -t "$S/large/")
for f in verdict.json stages.txt row.log commit.log build_summary.json target_family.json admission.json; do
  [ -f "$D/$f" ] && cp -al "$D/$f" "$S/records/"
done
find "$S" -type f \( -path '*/replay_bundle_p*' -o -path '*/replay_slim_p*' -o -name topology.json -o -name timeline.jsonl \
  -o -name timeline_report.json \) -printf "private %s %P\n" -delete >> "$EV/store_omitted.txt"
KINDS=${STORE_KINDS:-build records}
if [ -d "$S/records/match/capture" ] && [ "$(find "$S/records/match/capture" -type f | wc -l)" -gt "${CAPTURE_TAR_FILES:-100000}" ]; then
  rm -rf "$S/records/match/capture"; mkdir -p "$S/capture"
  tar cf "$S/capture/match-capture.tar" --exclude=topology.json -C "$D" match/capture
  [[ " $KINDS " == *" capture "* ]] || KINDS="$KINDS capture"
fi

meta() {  # meta KIND ROLE -> the tree's meta (serving-view's record-build layout), from the Build's own summaries
  "$VV" epoch store-meta "$D" --kind "$1" --role "$2" --rownum "$ROWNUM" --run-id "$RESEARCH_RUN_ID" --epoch-sha "$EPOCH_SHA" \
    --gpu "${GPU_NAME:-}" --pod "${POD:-}" --epoch "${EPOCH_NAME:-vllm-rebaseline-epoch}"
}

for kind in $KINDS; do
  case "$kind" in records) role=records;; large) role=records-large;; capture) role=records-capture;; *) role=programs;; esac
  meta "$kind" "$role" > "$EV/meta-$kind.json"
done
bash "$(dirname "$0")/store_trees.sh" "$S" $KINDS; rc=$?
rm -rf "$S"
exit $rc
