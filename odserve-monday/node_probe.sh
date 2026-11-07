#!/usr/bin/env bash
# node_probe.sh USER@HOST CMD: runs CMD on the host with the research key (research.pods.runpod's ssh_key, decoded from
# the VM's injected secret; in a shell that has lost the secret, the one key it already decoded under
# ~/.local/state/research/runpod) and prints "EXIT rc" after its output.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 3
cd "$MAIN_REPO" || exit 3
D=$HOME/.local/state/research/runpod
KEY=$("$PY" -c 'from research.pods.runpod import ssh_key; print(ssh_key())' 2> /dev/null) || {
  keys=("$D"/ssh-key-*)
  [ "${#keys[@]}" = 1 ] && [ -f "${keys[0]}" ] || { echo "REFUSED: no research ssh key (the secret RUNPOD_SSH_KEY_B64)"; exit 3; }
  KEY=${keys[0]}
}
KH=$("$PY" -c 'from research.pods.runpod import known_hosts; print(known_hosts())' 2> /dev/null) || KH=$D/known_hosts
echo "== $1"
ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o "UserKnownHostsFile=$KH" "$1" "$2"
echo "EXIT $?"
