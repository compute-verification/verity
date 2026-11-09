#!/usr/bin/env bash
# vllm-twostage: a read-only look at vy-challenger-1's gate from a peer holding lane vllm's gate key: its status and the
# pins log's last lines (the pinned tree the gate runs). Opens nothing, reserves nothing.
#   research run --on vy-nebius-cpu-3 --project verity --source TREE --cwd source --timeout 5m --send gate_probe.sh \
#     -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/gate_probe.sh'
set -uo pipefail
GATE=${GATE:-research@204.12.178.84}; KEY=${SSH_KEY:-$HOME/.ssh/vllm_lane_ed25519}
g() { ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$GATE" "$@"; }
echo "== status"; g status 2>&1 | cut -c1-600 | tail -n 20
echo "== pins (last 3)"; g pins 2>&1 | tail -n 3 | cut -c1-600
