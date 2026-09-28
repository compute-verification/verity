#!/bin/sh
# the steward service: restarted if it exits; the lease (--lease --standby) makes any second copy stand by.
# --reap judges staleness from origin's lane/* tips, so it needs the read-only deploy key (/data/state/secrets/verity-deploy)
# that `--fetch` uses: if watch.log shows FETCH-FAILED, take --reap out until the fetch works again.
cd /workspace/steward/verity
PY=/data/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
export PYTHONPATH="$(PYTHONPATH=tools/research/src "$PY" -m research env --source . | sed -n 's/^export PYTHONPATH=//p')"
export RESEARCH_REPO=/workspace/steward/verity RESEARCH_NOTES=/workspace/steward/research-notes RESEARCH_STORE=/workspace/steward/store
export RESEARCH_MACHINES_D=/workspace/steward/research-notes/machines.d GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8
while :; do
  "$PY" -s -m research notes watch --every 2 --pods --reap --custody-r2 --lease --standby --steward-id vy-control-verity \
    --liveness git --fetch --sync --stale-min 12 --since-hours 8 --idle-min 5 --exclude 'vllm*' --guardian-log none \
    >> /workspace/steward/watch.log 2>&1
  sleep 30
done
