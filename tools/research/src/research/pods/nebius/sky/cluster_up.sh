#!/usr/bin/env bash
# SkyPilot + Kueue on the Nebius server, run ON vy-nebius-1 as its research user once launch.sh and check.py have passed:
#
#   research pods sync vy-nebius-1                  # this checkout -> /workspace/src on the server
#   research pods ssh vy-nebius-1 -- bash /workspace/src/tools/research/src/research/pods/nebius/sky/cluster_up.sh --apply
#   (--plan instead of --apply prints the steps and changes nothing)
#
# 1. SkyPilot 0.13.0 (uv tool) and its API server as a systemd service, bound to 127.0.0.1:46580, usage collection off. Clients
#    reach it through an ssh tunnel, so the security group stays ssh-only.
# 2. The SSH Node Pool vy-nebius (`sky ssh up`): k3s on this host, reached over its private address with a pool key, plus the NVIDIA
#    GPU operator and its dcgm-exporter. The node is labelled verity.dev/server=<machine> (Kueue's flavor) and
#    skypilot.co/accelerator=rtxpro6000: SkyPilot has no canonical name for "RTX PRO 6000 Blackwell Server Edition", and without the
#    label a request for RTXPRO6000 matches nothing.
# 3. Kueue v0.19.6 with its plain-pod integration on (SkyPilot's pods are plain pods), then kueue.yaml.
# 4. ~/.sky/config.yaml from sky-config.yaml (when absent), and the usage sampler: a 5-minute systemd timer appending
#    usage_report.py's sample to /workspace/usage/queues.jsonl.
# Idempotent: each step checks first. It installs the files beside it (kueue.yaml, sky-config.yaml, usage_report.py).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:---plan}
case $mode in --plan|--apply) ;; *) echo "usage: cluster_up.sh --plan | --apply" >&2; exit 2 ;; esac
POOL=vy-nebius
CTX=ssh-$POOL
MACHINE=${VY_MACHINE:-vy-nebius-1}
SKY_VERSION=0.13.0
KUEUE=v0.19.6
SRC=$here
KUBECTL=(kubectl --context "$CTX")
IP=$( (ip -4 route get 1.1.1.1 2>/dev/null || true) | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
[ -n "$IP" ] || IP=$(hostname -I | awk '{print $1}')
step() { echo "== $*"; }
run() { if [ "$mode" = --apply ]; then "$@"; else echo "   would run: $*"; fi; }

[ -f /var/lib/vy-setup.done ] || { echo "not the Nebius server (no /var/lib/vy-setup.done)" >&2; exit 1; }
[ -f "$SRC/kueue.yaml" ] || { echo "no kueue.yaml beside $0: run it from a synced checkout" >&2; exit 1; }
echo "machine $MACHINE, private address $IP, pool $POOL, files from $SRC ($mode)"

step "1. SkyPilot $SKY_VERSION and its API server"
command -v sky > /dev/null && sky --version 2>/dev/null | grep -q "$SKY_VERSION" \
  || run uv tool install --force "skypilot[kubernetes]==$SKY_VERSION"
if [ ! -f /etc/systemd/system/sky-api.service ]; then
  unit="[Unit]
Description=SkyPilot API server (127.0.0.1:46580)
After=network-online.target

[Service]
User=$USER
Environment=HOME=$HOME SKYPILOT_DISABLE_USAGE_COLLECTION=1 PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$HOME/.local/bin/sky api start --host 127.0.0.1 --foreground
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target"
  if [ "$mode" = --apply ]; then printf '%s\n' "$unit" | sudo tee /etc/systemd/system/sky-api.service > /dev/null; else echo "   would write /etc/systemd/system/sky-api.service"; fi
fi
run sudo systemctl daemon-reload
run sudo systemctl enable --now sky-api.service

step "2. the SSH Node Pool $POOL"
[ -f ~/.ssh/vy-pool ] || run ssh-keygen -q -t ed25519 -N "" -C "vy-pool@$MACHINE" -f ~/.ssh/vy-pool
if [ -f ~/.ssh/vy-pool.pub ] && ! grep -qF "$(cat ~/.ssh/vy-pool.pub)" ~/.ssh/authorized_keys; then
  run sh -c 'cat ~/.ssh/vy-pool.pub >> ~/.ssh/authorized_keys'
fi
if ! grep -q "^$POOL:" ~/.sky/ssh_node_pools.yaml 2>/dev/null; then
  pool="$POOL:
  user: $USER
  identity_file: ~/.ssh/vy-pool
  hosts:
    - $IP"
  if [ "$mode" = --apply ]; then mkdir -p ~/.sky && printf '%s\n' "$pool" >> ~/.sky/ssh_node_pools.yaml; else echo "   would add to ~/.sky/ssh_node_pools.yaml:"; printf '%s\n' "$pool" | sed 's/^/     /'; fi
fi
"${KUBECTL[@]}" get nodes > /dev/null 2>&1 || run sky ssh up --infra "$POOL"
nodes=$("${KUBECTL[@]}" get nodes -o json 2>/dev/null || true)
node=$([ -n "$nodes" ] && printf '%s' "$nodes" | python3 -c '
import json, sys
for n in json.load(sys.stdin)["items"]:
    if any(a["address"] == sys.argv[1] for a in n["status"].get("addresses", [])):
        print(n["metadata"]["name"])' "$IP" || true)
if [ -n "$node" ] || [ "$mode" = --plan ]; then
  run "${KUBECTL[@]}" label node "${node:-<this node>}" "verity.dev/server=$MACHINE" skypilot.co/accelerator=rtxpro6000 --overwrite
else
  echo "no node with address $IP in $CTX" >&2; exit 1
fi

step "3. Kueue $KUEUE and the queues"
if ! "${KUBECTL[@]}" -n kueue-system get deployment kueue-controller-manager > /dev/null 2>&1; then
  run "${KUBECTL[@]}" apply --server-side -f "https://github.com/kubernetes-sigs/kueue/releases/download/$KUEUE/manifests.yaml"
  run "${KUBECTL[@]}" -n kueue-system rollout status deployment kueue-controller-manager --timeout=300s
fi
cfg=$("${KUBECTL[@]}" -n kueue-system get cm kueue-manager-config -o jsonpath='{.data.controller_manager_config\.yaml}' 2>/dev/null || true)
if ! printf '%s' "$cfg" | grep -q '^  - "pod"'; then
  if [ "$mode" = --apply ]; then
    printf '%s' "$cfg" | python3 -c '
import sys
s = sys.stdin.read()
assert "integrations:\n  frameworks:\n" in s, "unexpected Kueue config layout"
sys.stdout.write(s.replace("integrations:\n  frameworks:\n", "integrations:\n  frameworks:\n  - \"pod\"\n", 1))' > /tmp/kueue-config.yaml
    "${KUBECTL[@]}" -n kueue-system create cm kueue-manager-config --from-file=controller_manager_config.yaml=/tmp/kueue-config.yaml \
      --dry-run=client -o yaml | "${KUBECTL[@]}" -n kueue-system apply -f -
    "${KUBECTL[@]}" -n kueue-system rollout restart deployment kueue-controller-manager
    "${KUBECTL[@]}" -n kueue-system rollout status deployment kueue-controller-manager --timeout=300s
  else
    echo "   would add the plain-pod framework (\"pod\") to kueue-manager-config and restart Kueue"
  fi
fi
run "${KUBECTL[@]}" apply -f "$SRC/kueue.yaml"
"${KUBECTL[@]}" -n gpu-operator get svc dcgm-exporter > /dev/null 2>&1 && echo "   dcgm-exporter: present" \
  || echo "   dcgm-exporter: MISSING (the GPU operator's install in step 2 failed: see sky ssh up's log); usage_report samples without GPU utilization"

step "4. SkyPilot config and the usage sampler"
if [ ! -f ~/.sky/config.yaml ]; then
  run install -m 644 "$SRC/sky-config.yaml" ~/.sky/config.yaml
  run sudo systemctl restart sky-api.service
elif ! cmp -s "$SRC/sky-config.yaml" ~/.sky/config.yaml; then
  echo "   ~/.sky/config.yaml differs from sky-config.yaml: left as it is"
fi
run sudo install -m 755 "$SRC/usage_report.py" /usr/local/bin/vy-usage
run sudo install -d -o "$USER" -g "$USER" /workspace/usage
if [ ! -f /etc/systemd/system/vy-usage.timer ]; then
  svc="[Unit]
Description=sample Kueue queue usage into /workspace/usage/queues.jsonl

[Service]
Type=oneshot
ExecStart=/bin/sh -c '/usr/bin/python3 /usr/local/bin/vy-usage sample --kubectl \"k3s kubectl\" >> /workspace/usage/queues.jsonl'"
  timer="[Unit]
Description=sample Kueue queue usage every 5 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target"
  if [ "$mode" = --apply ]; then
    printf '%s\n' "$svc" | sudo tee /etc/systemd/system/vy-usage.service > /dev/null
    printf '%s\n' "$timer" | sudo tee /etc/systemd/system/vy-usage.timer > /dev/null
  else
    echo "   would write vy-usage.service and vy-usage.timer"
  fi
fi
run sudo systemctl daemon-reload
run sudo systemctl enable --now vy-usage.timer

step "done"
echo "   check:  sky check ssh && sky gpus list --infra ssh/$POOL && ${KUBECTL[*]} get clusterqueues"
echo "   client: ssh -N -L 46580:127.0.0.1:46580 $USER@<public ip>  then  sky api login -e http://127.0.0.1:46580"
