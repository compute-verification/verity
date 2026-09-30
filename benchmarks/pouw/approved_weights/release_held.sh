#!/usr/bin/env bash
# Put approved-weights fill jobs held in /workspace/pouw/approved-weights/held/ back in the queue once the jobs they were
# held for have drained (no PREFIX* job queued or running), or once the fill runner changes (its sha256 differs from
# RUNNER_SHA256: infra's fix for CPU jobs starting in filename order). A job is held by moving its script out of running/
# while it runs: the runner then doesn't requeue it after its chunk (it moves running/<job> only if the file exists).
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> -- \
#     bash <shipped tree>/benchmarks/pouw/approved_weights/release_held.sh RUNNER_SHA256 PREFIX
set -uo pipefail
SUM=$1; PREFIX=$2
FILL=/workspace/pouw/fill
HOLD=/workspace/pouw/approved-weights/held
RUNNER=/workspace/pouw/infra/bin/fill_runner.py
for _ in $(seq 1 1440); do
  why=""
  if [ "$(sha256sum "$RUNNER" | cut -d' ' -f1)" != "$SUM" ]; then
    why="the fill runner changed"
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
