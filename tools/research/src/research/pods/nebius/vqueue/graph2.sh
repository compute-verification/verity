#!/bin/bash
# graph2's CLI on the machine that holds it, for the verified queue's loop on a node (loop.sh passes --graph2 with this file):
# each argument quoted for the remote shell, over one shared ssh connection (ControlPersist) so a tick of calls pays one
# handshake. The machine is VQ_GRAPH2_NODE (default vy-nebius-cpu-7), a name the register resolves (`research pods ssh NODE
# --print`, the loop's own key and known hosts under its home), never an address.
set -u
node=${VQ_GRAPH2_NODE:-vy-nebius-cpu-7}
export PYTHONPATH=${PYTHONPATH:-$HOME/src/tools/research/src}
line=$("${VQ_PYTHON:-python3.12}" -m research pods ssh "$node" --print) || exit 1
eval "ssh=($line)"
q=$(printf " %q" "$@")
exec "${ssh[@]:0:${#ssh[@]}-1}" -o ControlMaster=auto -o "ControlPath=$HOME/.runpod/ssh/cm-g2-%C" -o ControlPersist=600 \
  "${ssh[-1]}" "sudo -n python3.12 /root/executor/infra/graph2.py --db /root/executor/graph2.db --events /root/executor/events$q"
