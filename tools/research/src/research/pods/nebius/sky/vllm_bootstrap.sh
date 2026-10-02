#!/usr/bin/env bash
# The vLLM bootstrap of a Nebius Kueue job, from the job's tree (job_tree.sh), skipped when it has already passed:
#
#   bash "$SRC/tools/research/src/research/pods/nebius/sky/vllm_bootstrap.sh" --cases "$ROLE" [--cpu] --out DIR
#
# pod_bootstrap.sh builds into one shared venv (/workspace/jobs/venv312), so it runs under one lock for the whole host, and jobs queued
# behind each other there (a 7-second capture waited 10-14 min behind GPU cells' ~3-min bootstraps, Sep 30 09:00-09:45Z). A bootstrap
# that passed leaves a stamp keyed by the job tree, the arguments and the venv's installed packages. A later job with the same key
# skips the bootstrap without taking the lock. The key is re-read inside the lock, since another job may have just finished it. A
# different tree, role or mode, or any change to the venv's packages, means a new key and a real bootstrap.
#
# A leased job's pod has no GPU of its own (GPU_LEASE set, dispatch.py's `lease: self`): a real GPU-mode bootstrap takes one from the
# lease, and takes it before the host-wide lock, never while holding it. A bootstrap that held the lock while it waited for a GPU
# stopped every other bootstrap on the host, CPU ones included, for as long as its pool had none to give: on vy-nebius-1 the circuits
# pool's holders queue in deployments-gpu behind the very GPU-stage pods whose bootstraps waited (Oct 1 22:13-23:02Z). Taken in this
# order, a holder of the lock waits for nothing but its own bootstrap; the cost is a leased GPU idle while its bootstrap waits for the
# lock, which the log says.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tree=$(cd "$here/../../../../../../.." && pwd)
work=${VY_JOBS:-/workspace/jobs}
stamps=$work/bootstrap/stamps
site=$work/venv312/lib/python3.12/site-packages
key() {
  { basename "$tree"; printf '%s\n' "$@"; ls "$site" 2>/dev/null; } | sha256sum | cut -c1-24
}
mkdir -p "$stamps"
k=$(key "$@")
if [ -e "$stamps/$k" ]; then
  echo "[bootstrap] cached: this tree, these arguments and this venv passed at $(cat "$stamps/$k") (stamp $k); skipped"
  echo BOOTSTRAP-OK
  exit 0
fi
if [ -n "${GPU_LEASE:-}" ] && [ -z "${VY_BOOTSTRAP_LEASED:-}" ] && [[ " $* " != *" --cpu "* ]]; then
  VY_BOOTSTRAP_LEASED=1 exec "$GPU_LEASE" 1 --wait --max-min 60 --mem-gb 0 -- bash "$here/vllm_bootstrap.sh" "$@"
fi
cd "$tree/integrations/vllm"
exec 9>> "$work/.vllm-bootstrap.lock"
if ! flock -n 9; then
  [ -z "${VY_BOOTSTRAP_LEASED:-}" ] || echo "[bootstrap] holding leased GPU ${CUDA_VISIBLE_DEVICES:-?}; waiting for the host-wide lock"
  flock 9
fi
k=$(key "$@")
if [ -e "$stamps/$k" ]; then
  echo "[bootstrap] cached while waiting (stamp $k); skipped"
  echo BOOTSTRAP-OK
  exit 0
fi
env -u REPO WORK=$work HF_HOME=${HF_HOME:-/workspace/hf} bash verity_vllm/ops/pod_bootstrap.sh "$@"
date -u +%Y-%m-%dT%H:%M:%SZ > "$stamps/$(key "$@")"
