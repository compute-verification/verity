#!/usr/bin/env bash
# Launches certside/job.sh (the network certifier's side of a served id-form run with a response cap) on a named CPU
# node as a `research run`, its tree this clean checkout's HEAD, from any host with research's credentials (the
# nodes' ssh key and the store's custody key), once the id-form endpoint is up on the serving node's loopback:
#   PORT=P [MODEL=NAME] [ID_BOUND=N] [the job's schedule and traffic variables] [MEM_GB=4] [TIMEOUT=90m]
#     [ON=vy-nebius-cpu-11] [MACHINE=vy-spot-glm53 | PEER=USER@HOST [SSH_PORT=N]] [KEY=FILE] bash certside/launch.sh [--reach]
# --reach runs the job's reach check alone (under a minute, waited on) and exits with its code: 0 reached, 3 not
# reached, 4 the key is accepted and the forward opens and nothing listens on PORT yet, 5 the model's name is too long.
# PORT is the endpoint's port on the serving node's 127.0.0.1, the one port the forward opens. MODEL, when the endpoint
# doesn't answer GET /v1/models, is its model's name: the reach check is then a connect, and the run sends it as
# --model. The job's variables (CELL_BYTES SLOTS BUCKET_MS DELAY LANES QUEUE WINDOW INGRESS_CELLS INGRESS_BUCKETS
# EGRESS_CELLS CAP RATE REQUESTS SEED PROMPT_MAX MAX_TOKENS TAIL ID_BOUND RECHECK) pass through when set here; their
# defaults are the job's, in its header. The serving node's tick's bucket must equal BUCKET_MS.
# PEER and SSH_PORT default to MACHINE's ssh endpoint as `research` reaches it: a RunPod pod's (node A, vy-spot-glm53)
# is root at its public IP and the public port RunPod maps to its 22/tcp, read from the RunPod API at launch (a pod's
# mapping is its own, so never written down); an ssh machine's is the registry's user, host and port. It reaches the
# job as --env PEER=…, in the run's launch record, and is printed nowhere: this script and the job name machines by
# their short names only. KEY is the tunnel key's path on the CPU node (the job's default ~/.ssh/vy-tick-tunnel).
# The run, not in a pool: a pool's slot runs its job at nice 19, behind the node's checks, which a ticking certifier
# can't be. MEM_GB reserved (waiting up to 5 min for it), TIMEOUT at most (about 23 min of serving at the defaults, then
# the check), its custody key 6 h. Everything it keeps is in its out/ (job.sh). Cite the run id in the report, which
# keeps its record in the store.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PORT=${PORT:?PORT: the endpoint port on 127.0.0.1 of the serving node}
ON=${ON:-vy-nebius-cpu-11}
R=(research); command -v research > /dev/null || R=(uv run -q research)
MACHINE=${MACHINE:-vy-spot-glm53}
REG=${RESEARCH_NOTES:-$HOME/.research/notes}/machines.d/$MACHINE.toml
field() { sed -n "s/^$1 = \"\{0,1\}\([^\"]*\)\"\{0,1\}\$/\1/p" "$REG" 2>/dev/null; }
if [ -z "${PEER:-}" ]; then
  case $(field provider) in
    runpod)
      pod=$(field pod_id)
      for _ in 1 2 3 4 5; do   # the REST API now and then answers a running pod with no IP or mapping (research.remote)
        read -r host mapped < <("${R[@]}" pods get "$pod" 2>/dev/null | python3 -c 'import json, sys; p = json.load(sys.stdin)
print(p.get("publicIp") or "-", (p.get("portMappings") or {}).get("22") or "-")' 2>/dev/null || echo "- -")
        [ "$host" != - ] && [ "$mapped" != - ] && break
        sleep 3
      done
      [ "$host" != - ] && [ "$mapped" != - ] || { echo "launch: the RunPod API gave no public IP and 22/tcp mapping" \
        "for $MACHINE's pod: is the pod running, and research's RunPod key set? Else PEER=USER@HOST SSH_PORT=N"; exit 2; }
      PEER=root@$host; SSH_PORT=${SSH_PORT:-$mapped};;
    ssh)
      host=$(field host); user=$(field user)
      [ -n "$host" ] || { echo "launch: no host for $MACHINE in the machines registry"; exit 2; }
      PEER=${user:-root}@$host; SSH_PORT=${SSH_PORT:-$(field port)};;
    *) echo "launch: no machine $MACHINE in the machines registry (MACHINE=NAME, or PEER=USER@HOST SSH_PORT=N)"; exit 2;;
  esac
  serving="the serving node $MACHINE"
else
  serving="the serving node (PEER as given)"
fi
envs=(--env "PEER=$PEER" --env "PORT=$PORT")
for name in SSH_PORT KEY MODEL ID_BOUND CELL_BYTES SLOTS BUCKET_MS DELAY LANES QUEUE WINDOW INGRESS_CELLS INGRESS_BUCKETS \
            EGRESS_CELLS CAP RATE REQUESTS SEED PROMPT_MAX MAX_TOKENS TAIL RECHECK; do
  [ -n "${!name:-}" ] && envs+=(--env "$name=${!name}")
done
run=("${R[@]}" run --on "$ON" --project verity --source "$PWD" --cwd source "${envs[@]}")
echo "launch: $serving's 127.0.0.1:$PORT from $ON, tree $(git rev-parse --short HEAD)${MODEL:+, model $MODEL}"
if [ "${1:-}" = --reach ]; then
  exec "${run[@]}" --env REACH_ONLY=1 --timeout 3m --wait -- bash certside/job.sh
fi
exec "${run[@]}" --mem-gb "${MEM_GB:-4}" --mem-wait 300 --timeout "${TIMEOUT:-90m}" --custody-ttl 6h -- bash certside/job.sh
