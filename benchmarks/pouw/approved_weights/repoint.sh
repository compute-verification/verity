#!/usr/bin/env bash
# Repoint queued approved-weights fill jobs at a newer shipped tree, keeping their output directories (the checkpoint
# format must be unchanged between the two trees). Each job is repointed once, between chunks: it is taken out of the queue
# by a rename (which fails if the runner has just started it), edited, and put back. Waits up to --hours for each job.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> -- \
#     bash <shipped tree>/benchmarks/pouw/approved_weights/repoint.sh OLD_SHA JOB [JOB ...]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NEW=$(basename "$(cd "$HERE/../../.." && pwd)")
OLD=$1; shift
FILL=/workspace/pouw/fill
HOLD=/workspace/pouw/approved-weights/repoint-hold
mkdir -p "$HOLD"
left=("$@")
for _ in $(seq 1 7200); do
  next=()
  for j in "${left[@]}"; do
    if [ -e "$FILL/done/$j" ] || [ -e "$FILL/failed/$j" ]; then
      echo "$j finished before it was repointed"; continue
    fi
    if mv "$FILL/queue/$j" "$HOLD/$j" 2>/dev/null; then
      sed -i "s#/src/$OLD/#/src/$NEW/#g" "$HOLD/$j"
      mv "$HOLD/$j" "$FILL/queue/$j"
      echo "$j repointed $OLD -> $NEW at $(date -u +%FT%TZ)"
    else
      next+=("$j")
    fi
  done
  left=("${next[@]}")
  [ ${#left[@]} -eq 0 ] && exit 0
  sleep 1
done
echo "not repointed: ${left[*]}"
exit 1
