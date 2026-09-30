#!/usr/bin/env bash
# Cutover on vy-nebius-1: every GPU to Kueue (provers back to 4), once no direct GPU work is left. Run on the server:
#
#   bash cutover.sh [--wait-hours H]     (default 12)
#
# It interrupts nothing. It waits (polling every minute, up to H hours) until no gpu-lease lock is held and no GPU process runs
# outside a Kubernetes pod, then runs `VY_DIRECT_GPUS=none cluster_up.sh --apply`. That writes /etc/vy/direct-gpus = none, so
# gpu-lease refuses from then on, deletes the placeholder pod, and applies kueue.yaml with provers' 4 GPUs. Running Kueue jobs
# keep running. Exit 3 if the wait runs out.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
hours=12
[ "${1:-}" = --wait-hours ] && hours=${2:?}

direct_work() {
  # prints what still holds a GPU outside Kubernetes; nothing when clear
  local i p
  for i in $(seq 0 7); do
    [ -e "/run/gpu-lease/$i.lock" ] || continue
    flock -n -x "/run/gpu-lease/$i.lock" true 2> /dev/null || echo "gpu-lease lock $i: $(cat "/run/gpu-lease/$i.owner" 2> /dev/null)"
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null); do
    grep -q kubepods "/proc/$p/cgroup" 2> /dev/null || echo "GPU process $p outside Kubernetes: $(tr '\0' ' ' < "/proc/$p/cmdline" | cut -c1-120)"
  done
}

deadline=$(( $(date +%s) + hours * 3600 ))
while :; do
  left=$(direct_work)
  [ -z "$left" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "cutover: still waiting after ${hours} h:"; echo "$left"; exit 3
  fi
  echo "$(date -u +%H:%MZ) cutover waits for: $(echo "$left" | tr '\n' ';')"
  sleep 60
done
echo "$(date -u +%H:%MZ) no direct GPU work left: cutting over"
VY_DIRECT_GPUS=none bash "$here/cluster_up.sh" --apply
echo "$(date -u +%H:%MZ) cutover done: /etc/vy/direct-gpus = $(cat /etc/vy/direct-gpus)"
