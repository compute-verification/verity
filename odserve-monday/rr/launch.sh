#!/usr/bin/env bash
# launch.sh NODE [KEY=VALUE ...]: remote_run.sh on NODE in a clone of the run branch, its inputs sent from this directory.
# Node-parametric: NODE is the research machine and the gate's peer name (H1_PEER) unless H1_PEER is given; H1_ROOT (the
# registration's weights root prefix under circuit.txt's Program), SETTINGS (the node's private load file) and SOURCE (the
# run branch's checkout here) come from the environment. KEY=VALUE pairs reach remote_run.sh and served_verdicts.sh.
set -euo pipefail
NODE=${1:?the node}; shift
H=${H:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
# A store run id (r<date>-<time>-<4 hex>): the runner's custody publishes no other form, and its .custody then says preserved false.
RID=${RID:-r$(date -u +%Y%m%d-%H%M%S)-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')}
[[ $RID =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || { echo "REFUSED: RID=$RID is not a store run id (r<date>-<time>-<4 hex>)" >&2; exit 3; }
: "${H1_ROOT:?the first hex digits of H1_MODEL's registered weights root under $(head -c 16 "$H/circuit.txt")... (the gate's models)}"
sends=()
for f in remote_run.sh main.txt stated.txt scrub.sh scan-out.py retrace-row.sh; do sends+=(--send "$H/$f"); done
[ -s "$H/keys/challenger-key.json" ] && sends+=(--send "$H/keys/challenger-key.json")
envs=(--env "H1_PEER=${H1_PEER:-$NODE}" --env "GATE_PIN=$(head -1 "$H/main.txt")" --env "H1_CIRCUIT=$(cat "$H/circuit.txt")"
      --env "H1_ROOT=$H1_ROOT" --env "SETTINGS=${SETTINGS:-/workspace/research/vllm-served-verdicts/settings-remote/settings.env}")
for kv in "$@"; do envs+=(--env "$kv"); done
echo "$RID"
exec research run --on "$NODE" --project verity --source "${SOURCE:?the run branch checkout to ship}" --cwd clone \
  --timeout "${TIMEOUT:-18h}" --custody-ttl "${CUSTODY_TTL:-20h}" --id "$RID" --declared-output 'out/*' \
  "${sends[@]}" "${envs[@]}" -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/remote_run.sh"'
