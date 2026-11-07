#!/usr/bin/env bash
# Puts network-accounting's 650 GB memory hold back on the follow-on node, from infra's master copy
# ~research/netacct-inf458-thu.json, about an hour before the follow-on; follow.sh's launch removes it as the job starts
# with its own --mem-gb. Prints PLACED, ALREADY HELD or REFUSED, and the node's memory.
#   [FOLLOW_NODE=vy-nebius-cpu-11] bash place-hold.sh
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
IP=$(follow_ip "$FOLLOW_NODE") || exit 3
bash "$H/node_probe.sh" "research@$IP" 'm=$HOME/netacct-inf458-thu.json f=/workspace/ramlock/netacct-inf458-thu.json; if [ -e "$f" ]; then echo "ALREADY HELD $f"; elif grep -q "\"owner\": \"network-accounting\"" "$m" && grep -q "\"gb\": 650" "$m"; then cp "$m" "$f" && echo "PLACED $f at $(date -u +%H:%M:%SZ): $(head -c 120 "$f")"; else echo "REFUSED: $m is not the 650 GB hold of network-accounting"; fi; free -g | sed -n 2p; df -h /workspace | tail -1'
