#!/usr/bin/env bash
# E8 NIGHT 2026-09-12 -- pod-side row writer (D37: the laptop may be dead when the run ends).
# Watches orchestrate_32x16_pod.sh's log; writes the night-table row of THIS run into the record dir (mirrored by the laptop puller) and into
# the pod tree's out/night/table/ with e2e.demo_core.night.row_from_record:
#   * at 12:30Z, if the run is not finished: the row with `prove_s: NOT RUN (running since <utc of phase B>)` and every other field MEASURED so far;
#   * when the orchestrator logs `DONE:`: the final row, with reverify_clean.json.
# Usage (pod, nohup + setsid): row_watch_pod.sh <tag> <prompt_len> <max_tokens> [publish_hhmm=12:30]
set -u
TAG=${1:?tag}; LP=${2:?prompt_len}; MT=${3:?max_tokens}; PUB=${4:-12:30}
RUN_ID=${RUN_ID:-V5-N${LP}x${MT}-1req-bex0-inc}
REPO=${REPO:-/workspace/a9/veritor}; PY=$REPO/.venv/bin/python
REC=/workspace/a9/rec/$RUN_ID-$TAG; LOG=/workspace/a9/rec/orchestrate-$TAG.log
POD_SINCE=${POD_SINCE:-2026-09-12T05:28:32Z}; RATE=${POD_USD_PER_H:-0.74}
# [D40] ROW_EXTRA_FILE: one extra row_from_record argument per line (--pod-name, --extra-cost-usd, --note "..." with spaces), read into an array
ROW_EXTRA=(); if [ -n "${ROW_EXTRA_FILE:-}" ] && [ -f "$ROW_EXTRA_FILE" ]; then while IFS= read -r l; do [ -n "$l" ] && ROW_EXTRA+=("$l"); done < "$ROW_EXTRA_FILE"; fi
export PYTHONPATH=$REPO:$REPO/vllm-poc:$REPO/src
row() {  # $1 = extra args
  # shellcheck disable=SC2086
  # --dry prints the row JSON (-> row.json); the second call writes the table file (its stdout is a one-line summary, not the row -- the n2 row.json
  # on the pod carried that summary line: fixed here after the fact, the row was regenerated on the laptop from the slim archive)
  (cd "$REPO" && $PY -m e2e.demo_core.night.row_from_record --record "$REC" --prompt-len "$LP" --max-tokens "$MT" --pod-since "$POD_SINCE" \
     --pod-usd-per-h "$RATE" --record-path "$REC" --table "$REPO/out/night/table" $1 --dry "${ROW_EXTRA[@]}") > "$REC/row.json.tmp" 2> "$REC/row.err" && mv "$REC/row.json.tmp" "$REC/row.json"
  (cd "$REPO" && $PY -m e2e.demo_core.night.row_from_record --record "$REC" --prompt-len "$LP" --max-tokens "$MT" --pod-since "$POD_SINCE" \
     --pod-usd-per-h "$RATE" --record-path "$REC" --table "$REPO/out/night/table" $1 "${ROW_EXTRA[@]}") >> "$REC/row.err" 2>&1
  cp "$REPO/out/night/table/SmolLM2-135M__${LP}x${MT}.json" "$REC/table-row-SmolLM2-135M__${LP}x${MT}.json" 2>/dev/null
  [ -n "${VOL:-}" ] && { cp "$REC/row.json" "$VOL/rec/row-$TAG.json" 2>/dev/null; cp "$REPO/out/night/table/SmolLM2-135M__${LP}x${MT}.json" "$VOL/rec/table-row-SmolLM2-135M__${LP}x${MT}-$TAG.json" 2>/dev/null; }
  echo "[row_watch $(date -u +%FT%TZ)] wrote row ($1): $(head -c 200 "$REC/row.json" | tr '\n' ' ')" >> "$LOG"
}
published=
while ! grep -q '^\[orch [^]]*\] DONE:' "$LOG" 2>/dev/null; do
  now=$(date -u +%H:%M)
  if [ -z "$published" ] && [[ "$now" > "$PUB" ]] && [ -f "$REC/freeze.json" ]; then
    since=$(grep -o '^\[orch [^]]*\] phase B' "$LOG" | head -1 | sed -E 's/^\[orch ([^]]*)\].*/\1/')
    row "--running-since ${since:-$(date -u +%FT%TZ)}"; published=1
  fi
  sleep 60
done
RV=/workspace/a9/reverify/$TAG/reverify_clean.json
row "$([ -f "$RV" ] && echo "--reverify $RV")"
echo "[row_watch $(date -u +%FT%TZ)] final row written: $REC/row.json" >> "$LOG"
