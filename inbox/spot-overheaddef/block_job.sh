#!/usr/bin/env bash
# Inside gpu-lease's grant of GPUs 0-7 on vy-spot-glm53: the whole-block replay with every hook on (spot_canonical.sh, its settings
# from the environment), started only if the grant came by START_BY, when one is set.
set -u
echo "GRANTED $(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} GPU_LEASE_UUID=${GPU_LEASE_UUID-unset} DISK $(df -BG --output=avail /workspace | tail -1)"
if [ -n "${START_BY:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$START_BY" +%s)" ]; then
  echo "too late: granted after $START_BY, so it can't end in time"
  exit 6
fi
exec bash "$RESEARCH_RUN_DIR/inputs/spot_canonical.sh"
