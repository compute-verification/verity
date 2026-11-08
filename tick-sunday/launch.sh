#!/usr/bin/env bash
# Launches tick-sunday/job.sh (the certified tick's network certifier side) on vy-nebius-cpu-11 as a `research run`, its
# tree this clean checkout's HEAD, from any host with research's credentials (the nodes' ssh key and the store's custody
# key), once vllm's endpoint is up on the serving node's loopback:
#   PORT=P [PEER=research@HOST] [KEY=FILE] [RATE=1.67] [REQUESTS=3000] bash tick-sunday/launch.sh [--reach]
# --reach runs the job's reach check alone (about a minute, waited on): run it before the slot. PORT is the endpoint's port
# on the serving node's 127.0.0.1; PEER defaults to vy-dc-h200's address in the machine registry; KEY is the tunnel key's
# path on cpu-11 (the job's default ~/.ssh/vy-tick-tunnel).
# The run, not in a pool: a pool's slot runs its job at nice 19, behind the node's checks, which a ticking certifier can't
# be. 4 GB reserved (waiting up to 5 min for it), 90 min at most (about 36 min of serving and checking), its custody key
# 6 h. Everything it keeps is in its out/ (job.sh). Cite the run id in the report, which keeps its record in the store.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PORT=${PORT:?PORT: the endpoint port on 127.0.0.1 of the serving node}
REG=${RESEARCH_NOTES:-$HOME/.research/notes}/machines.d/vy-dc-h200.toml
if [ -z "${PEER:-}" ]; then
  host=$(sed -n 's/^host = "\(.*\)"$/\1/p' "$REG" 2>/dev/null || true)
  [ -n "$host" ] || { echo "launch: no host in $REG (PEER=research@HOST)"; exit 2; }
  PEER=research@$host
fi
envs=(--env "PEER=$PEER" --env "PORT=$PORT" --env "RATE=${RATE:-1.67}" --env "REQUESTS=${REQUESTS:-3000}"
      --env "SEED=${SEED:-0}")
[ -n "${KEY:-}" ] && envs+=(--env "KEY=$KEY")
run=(research run --on vy-nebius-cpu-11 --project verity --source "$PWD" --cwd source "${envs[@]}")
echo "launch: $PEER's 127.0.0.1:$PORT from vy-nebius-cpu-11, tree $(git rev-parse --short HEAD)"
if [ "${1:-}" = --reach ]; then
  exec "${run[@]}" --env REACH_ONLY=1 --timeout 3m --wait -- bash tick-sunday/job.sh
fi
exec "${run[@]}" --mem-gb 4 --mem-wait 300 --timeout 90m --custody-ttl 6h -- bash tick-sunday/job.sh
