#!/usr/bin/env bash
# Put approved-weights fill jobs held in /workspace/pouw/approved-weights/held/ back in the queue once the jobs they were
# held for have drained (no PREFIX* job queued or running), or once someone creates held/release (when infra posts the
# fix for CPU jobs starting in filename order; any edit to the runner is not a sign of it). A job is held by moving its
# script out of running/ while it runs: the runner then doesn't requeue it after its chunk (it moves running/<job> only if
# the file exists).
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> -- \
#     bash <shipped tree>/benchmarks/pouw/approved_weights/release_held.sh PREFIX
set -uo pipefail
PREFIX=$1
FILL=/workspace/pouw/fill
HOLD=/workspace/pouw/approved-weights/held
for _ in $(seq 1 1440); do
  why=""
  if [ -e "$HOLD/release" ]; then
    why="$HOLD/release exists"
  elif ! ls "$FILL/queue" "$FILL/running" | grep -q "^$PREFIX"; then
    why="no $PREFIX* job is queued or running"
  fi
  if [ -n "$why" ]; then
    for p in "$HOLD"/*.sh; do
      [ -e "$p" ] || continue
      mv "$p" "$FILL/queue/$(basename "$p")"
      echo "$(basename "$p") released at $(date -u +%FT%TZ): $why" | tee -a "$HOLD/hold.log"
    done
    exit 0
  fi
  sleep 30
done
echo "still held after 12 h: $(ls "$HOLD"/*.sh 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
exit 1
