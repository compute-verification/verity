#!/usr/bin/env bash
# Launches tick-sunday/job.sh (the certified tick's network certifier side) on vy-nebius-cpu-11 as a `research run`, its
# tree this clean checkout's HEAD, from any host with research's credentials (the nodes' ssh key and the store's custody
# key), once vllm's endpoint is up on the serving node's loopback:
#   PORT=P [MACHINE=vy-spot-glm53 | PEER=USER@HOST [SSH_PORT=N]] [KEY=FILE] [RATE=1.67] [REQUESTS=3000] \
#     bash tick-sunday/launch.sh [--reach]
# --reach runs the job's reach check alone (about a minute, waited on): run it before the slot. PORT is the endpoint's port
# on the serving node's 127.0.0.1. PEER and SSH_PORT default to MACHINE's ssh endpoint as `research` reaches it: a RunPod
# pod's (node A, vy-spot-glm53) is root at its public IP and the public port RunPod maps to its 22/tcp, read from the RunPod
# API at launch (a pod's mapping is its own, so never written down); an ssh machine's is the registry's user, host and port.
# KEY is the tunnel key's path on cpu-11 (the job's default ~/.ssh/vy-tick-tunnel).
# The run, not in a pool: a pool's slot runs its job at nice 19, behind the node's checks, which a ticking certifier can't
# be. 4 GB reserved (waiting up to 5 min for it), 90 min at most (about 36 min of serving and checking), its custody key
# 6 h. Everything it keeps is in its out/ (job.sh). Cite the run id in the report, which keeps its record in the store.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PORT=${PORT:?PORT: the endpoint port on 127.0.0.1 of the serving node}
MACHINE=${MACHINE:-vy-spot-glm53}
REG=${RESEARCH_NOTES:-$HOME/.research/notes}/machines.d/$MACHINE.toml
field() { sed -n "s/^$1 = \"\{0,1\}\([^\"]*\)\"\{0,1\}\$/\1/p" "$REG" 2>/dev/null; }
if [ -z "${PEER:-}" ]; then
  case $(field provider) in
    runpod)
      pod=$(field pod_id)
      for _ in 1 2 3 4 5; do   # the REST API now and then answers a running pod with no IP or mapping (research.remote)
        read -r host mapped < <(research pods get "$pod" 2>/dev/null | python3 -c 'import json, sys; p = json.load(sys.stdin)
print(p.get("publicIp") or "-", (p.get("portMappings") or {}).get("22") or "-")' 2>/dev/null || echo "- -")
        [ "$host" != - ] && [ "$mapped" != - ] && break
        sleep 3
      done
      [ "$host" != - ] && [ "$mapped" != - ] || { echo "launch: \`research pods get $pod\` ($MACHINE) gave no public IP" \
        "and 22/tcp mapping: is the pod running, and research's RunPod key set? Else PEER=USER@HOST SSH_PORT=N"; exit 2; }
      PEER=root@$host; SSH_PORT=${SSH_PORT:-$mapped};;
    ssh)
      host=$(field host); user=$(field user)
      [ -n "$host" ] || { echo "launch: no host in $REG"; exit 2; }
      PEER=${user:-root}@$host; SSH_PORT=${SSH_PORT:-$(field port)};;
    *) echo "launch: no machine $MACHINE in $REG (MACHINE=NAME, or PEER=USER@HOST SSH_PORT=N)"; exit 2;;
  esac
fi
envs=(--env "PEER=$PEER" --env "PORT=$PORT" --env "RATE=${RATE:-1.67}" --env "REQUESTS=${REQUESTS:-3000}"
      --env "SEED=${SEED:-0}")
[ -n "${KEY:-}" ] && envs+=(--env "KEY=$KEY")
[ -n "${SSH_PORT:-}" ] && envs+=(--env "SSH_PORT=$SSH_PORT")
run=(research run --on vy-nebius-cpu-11 --project verity --source "$PWD" --cwd source "${envs[@]}")
echo "launch: $PEER (ssh port ${SSH_PORT:-22})'s 127.0.0.1:$PORT from vy-nebius-cpu-11, tree $(git rev-parse --short HEAD)"
if [ "${1:-}" = --reach ]; then
  exec "${run[@]}" --env REACH_ONLY=1 --timeout 3m --wait -- bash tick-sunday/job.sh
fi
exec "${run[@]}" --mem-gb 4 --mem-wait 300 --timeout 90m --custody-ttl 6h -- bash tick-sunday/job.sh
