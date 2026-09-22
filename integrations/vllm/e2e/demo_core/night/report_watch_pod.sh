#!/usr/bin/env bash
# E8 NIGHT 2026-09-12 -- pod-side REPORT-e8.md refresher (D37): rewrites /workspace/REPORT-e8.md every 5 min while the run is in progress and once more
# after the orchestrator's DONE line (+ after the row watcher's final row). Usage: report_watch_pod.sh <tag>
TAG=${1:?tag}; REPO=${REPO:-/workspace/a9/veritor}
LOG=/workspace/a9/rec/orchestrate-$TAG.log
export PYTHONPATH=$REPO:$REPO/vllm-poc:$REPO/src
# [D40] REPORT_ARGS: extra report_pod args (--pod-line ... --archive-dir ... --extra-out /vol/e8/REPORT-e8.md); the volume copy is the durable one
run() { (cd /workspace/a9 && eval $REPO/.venv/bin/python -m e2e.demo_core.night.report_pod --tag "$TAG" --out /workspace/REPORT-e8.md ${REPORT_ARGS:-} >> /workspace/a9/rec/report_watch-$TAG.log 2>&1); }
while ! grep -q '] DONE:' "$LOG" 2>/dev/null; do run; sleep 300; done
run; sleep 120; run   # once after DONE, once after the row watcher's final row
echo "[report_watch $(date -u +%FT%TZ)] final /workspace/REPORT-e8.md" >> "$LOG"
