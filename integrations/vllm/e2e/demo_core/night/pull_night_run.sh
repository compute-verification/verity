#!/usr/bin/env bash
# [E8 NIGHT 2026-09-12] Laptop-side puller for the 32x16 run (the -03 archive discipline: MERGE into the local mirror, never replace; rsync
# without --delete), beside E10's record_v5/pull_checkpoints.sh idea but by SIZE: every file of the record dir <= MAX (default 8 MB) is
# mirrored -- freeze/instance/coverage/serving/chronology/protocol/verify/obligations/*.json, logs, composed.json -- rows, trees, leaf/value
# blobs, witnesses and proofs (large) are NOT (they come with the slim archive).  Also mirrors the launch / orchestrate logs and freeze-<tag>.json.
# usage: nohup bash pull_night_run.sh <ssh-wrapper> <tag> <local dir> [interval_s=120] [max=8m] > <local dir>.pull.log 2>&1 &
set -u
SSHW=${1:?ssh wrapper}; TAG=${2:?tag}; LOCAL=${3:?local dir}; EVERY=${4:-120}; MAX=${5:-8m}
RUN_ID=${RUN_ID:-V5-N32x16-1req-bex0-inc}
REMOTE=/workspace/a9/rec/$RUN_ID-$TAG
SSHLINE=$(grep -E '^exec ssh|^ssh' "$SSHW" | head -1 | sed -E 's/^exec //; s/ "\$@"//')
RHOST=$(echo "$SSHLINE" | awk '{print $NF}')
SSHE=$(echo "$SSHLINE" | sed -E "s/ $RHOST\$//")
mkdir -p "$LOCAL" "$LOCAL/../pod-logs-$TAG"
stamp() { date -u +%FT%TZ; }
misses=0; n=0
while :; do
  n=$((n + 1))
  if rsync -az --timeout=30 --max-size="$MAX" -e "$SSHE" "$RHOST:$REMOTE/" "$LOCAL/" 2>"$LOCAL/.pull.err"; then
    misses=0
    rsync -az --timeout=30 -e "$SSHE" "$RHOST:/workspace/a9/rec/launch-$TAG.out" "$RHOST:/workspace/a9/rec/launch-$TAG.txt" "$RHOST:/workspace/a9/rec/orchestrate-$TAG.log" "$RHOST:/workspace/a9/rec/orchestrate-$TAG.out" "$RHOST:/workspace/a9/rec/freeze-$TAG.json" "$RHOST:/workspace/a9/rec/inputs-$TAG.sha256" "$RHOST:/workspace/a9/rec/$RUN_ID-$TAG.out" "$LOCAL/../pod-logs-$TAG/" 2>/dev/null
    rsync -az --timeout=60 --max-size=64m -e "$SSHE" "$RHOST:/workspace/a9/reverify/$TAG/" "$LOCAL/../reverify-$TAG/" 2>/dev/null
    rsync -az --timeout=60 -e "$SSHE" --include='*.sha256' --include='*-slim.tgz' --exclude='*' "$RHOST:/workspace/a9/archive/" "$LOCAL/../archive/" 2>/dev/null
    echo "[pull] $(stamp) poll $n: mirrored ($(du -sk "$LOCAL" | cut -f1) KB) $( [ -f "$LOCAL/chronology/selected.json" ] && echo 'selected.json PRESENT' || echo '(no sample yet)') $( [ -f "$LOCAL/verify/verify_v5.json" ] && echo 'verify_v5.json PRESENT')"
    if grep -q "DONE: archive" "$LOCAL/../pod-logs-$TAG/orchestrate-$TAG.log" 2>/dev/null; then echo "[pull] $(stamp) orchestrator DONE; final mirror done"; exit 0; fi
  else
    misses=$((misses + 1))
    echo "[pull] $(stamp) poll $n: rsync failed ($misses/10): $(tail -1 "$LOCAL/.pull.err" | cut -c1-120)"
    [ "$misses" -ge 10 ] && { echo "[pull] $(stamp) pod not answering; what is mirrored under $LOCAL is what survives"; exit 3; }
  fi
  sleep "$EVERY"
done
