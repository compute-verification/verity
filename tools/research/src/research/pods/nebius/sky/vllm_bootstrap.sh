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
cd "$tree/integrations/vllm"
exec 9>> "$work/.vllm-bootstrap.lock"
flock 9
k=$(key "$@")
if [ -e "$stamps/$k" ]; then
  echo "[bootstrap] cached while waiting (stamp $k); skipped"
  echo BOOTSTRAP-OK
  exit 0
fi
env -u REPO WORK=$work HF_HOME=${HF_HOME:-/workspace/hf} bash verity_vllm/ops/pod_bootstrap.sh "$@"
date -u +%Y-%m-%dT%H:%M:%SZ > "$stamps/$(key "$@")"
