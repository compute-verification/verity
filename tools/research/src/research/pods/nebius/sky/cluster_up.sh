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
# Another server runs its own single-node cluster from the same code, e.g. the PoUW server:
#   VY_MACHINE=vy-nebius-2 VY_POOL=vy-nebius-2 VY_QUEUES=kueue-pouw.yaml VY_DIRECT_GPUS=none cluster_up.sh --apply
set -euo pipefail
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH KUBECONFIG=${KUBECONFIG:-$HOME/.kube/config}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:---plan}
case $mode in --plan|--apply) ;; *) echo "usage: cluster_up.sh --plan | --apply" >&2; exit 2 ;; esac
POOL=${VY_POOL:-vy-nebius}
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
QUEUES=$SRC/${VY_QUEUES:-kueue.yaml}
[ -f "$QUEUES" ] || { echo "no $QUEUES beside $0: run it from a synced checkout" >&2; exit 1; }
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
Environment=HOME=$HOME KUBECONFIG=$HOME/.kube/config SKYPILOT_DISABLE_USAGE_COLLECTION=1 PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$HOME/.local/bin/sky api start --host 127.0.0.1 --foreground
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target"
  if [ "$mode" = --apply ]; then printf '%s\n' "$unit" | sudo tee /etc/systemd/system/sky-api.service > /dev/null; else echo "   would write /etc/systemd/system/sky-api.service"; fi
fi
run sudo systemctl daemon-reload
run sudo systemctl enable --now sky-api.service
[ "$mode" = --plan ] || for _ in $(seq 60); do curl -sf http://127.0.0.1:46580/api/health > /dev/null && break; sleep 2; done
command -v socat > /dev/null || run sudo apt-get install -y -q socat    # sky's Kubernetes port-forward mode needs it

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

[ -n "${VY_DIRECT_GPUS:-}" ] || VY_DIRECT_GPUS=$(cat /etc/vy/direct-gpus 2>/dev/null || echo 4,5,6,7)
step "2b. GPUs kept for direct runs: $VY_DIRECT_GPUS (nvidia-smi indices; none: all GPUs to Kubernetes)"
# While lanes still run GPU work directly (research run --on, outside Kubernetes), a placeholder pod, vy-system/direct-gpus, holds
# those GPUs through the device plugin, so no job is scheduled onto them. The plugin picks the devices, so the pod is made until
# one holds exactly VY_DIRECT_GPUS (the plugin hands the next pod the rest). Device minors are not nvidia-smi indices here
# (index 0 is /dev/nvidia7): the check maps each through `nvidia-smi -q`. Cutover: VY_DIRECT_GPUS=none deletes the holder.
held() {   # the nvidia-smi indices pod $1 holds, sorted, comma-joined
  local d m i out=""
  for d in $("${KUBECTL[@]}" -n vy-system exec "$1" -- ls /dev 2>/dev/null | grep -E '^nvidia[0-9]+$'); do
    m=${d#nvidia}
    for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      [ "$(nvidia-smi -q -i "$i" | awk '/Minor Number/ {print $NF}')" = "$m" ] && out="$out $i"
    done
  done
  echo "$out" | tr ' ' '\n' | grep . | sort -n | paste -sd, -
}
holder() {   # holder NAME COUNT: a pod holding COUNT GPUs, not managed by Kueue
  "${KUBECTL[@]}" apply -f - > /dev/null <<Y
apiVersion: v1
kind: Pod
metadata: {name: $1, namespace: vy-system, labels: {app: direct-gpus}}
spec:
  priorityClassName: system-node-critical
  runtimeClassName: nvidia
  terminationGracePeriodSeconds: 1
  containers:
  - {name: hold, image: "busybox:1.36", command: [sh, -c, "sleep 2147483647"], resources: {limits: {nvidia.com/gpu: $2}}}
Y
  "${KUBECTL[@]}" -n vy-system wait --for=condition=Ready "pod/$1" --timeout=180s > /dev/null
}
if [ "$VY_DIRECT_GPUS" = none ]; then
  run "${KUBECTL[@]}" -n vy-system delete pod direct-gpus direct-gpus-try --ignore-not-found
elif [ "$mode" = --plan ]; then
  echo "   would hold GPUs $VY_DIRECT_GPUS in vy-system/direct-gpus (now: $(held direct-gpus 2>/dev/null || echo none))"
elif [ "$(held direct-gpus)" != "$VY_DIRECT_GPUS" ] && [ "$(held direct-gpus-try 2>/dev/null)" != "$VY_DIRECT_GPUS" ]; then
  n=$(echo "$VY_DIRECT_GPUS" | tr ',' '\n' | grep -c .)
  "${KUBECTL[@]}" create namespace vy-system --dry-run=client -o yaml | "${KUBECTL[@]}" apply -f - > /dev/null
  "${KUBECTL[@]}" -n vy-system delete pod direct-gpus direct-gpus-try --ignore-not-found --wait=true > /dev/null
  holder direct-gpus "$n"
  if [ "$(held direct-gpus)" != "$VY_DIRECT_GPUS" ]; then
    holder direct-gpus-try "$n"
    "${KUBECTL[@]}" -n vy-system delete pod direct-gpus --wait=true > /dev/null
    [ "$(held direct-gpus-try)" = "$VY_DIRECT_GPUS" ] || { echo "   the device plugin gave neither holder GPUs $VY_DIRECT_GPUS" >&2; exit 1; }
    echo "   holding GPUs $VY_DIRECT_GPUS in vy-system/direct-gpus-try"
  fi
fi
# gpu-lease hands out only these (and nothing once they're `none`), so a direct lease never lands on a GPU Kubernetes owns.
if [ "$mode" = --apply ]; then echo "$VY_DIRECT_GPUS" | sudo tee /etc/vy/direct-gpus > /dev/null; else echo "   would write /etc/vy/direct-gpus: $VY_DIRECT_GPUS"; fi
run sudo install -m 755 "$SRC/../../sh/gpu_lease.sh" /usr/local/bin/gpu-lease
for p in direct-gpus direct-gpus-try; do
  h=$(held "$p" 2>/dev/null || true)
  [ -n "$h" ] && echo "   vy-system/$p holds GPUs $h for direct runs"
done
echo "   Kubernetes may allocate: $("${KUBECTL[@]}" describe node "${node:-x}" 2>/dev/null | awk '/Allocated resources/,0' | awk '$1 == "nvidia.com/gpu" {print 8 - $2}') of 8 GPUs"

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
# kueue.yaml holds the target split (provers 2 GPUs); provers gives up one GPU per GPU held for direct runs.
direct=$([ "$VY_DIRECT_GPUS" = none ] && echo 0 || echo "$VY_DIRECT_GPUS" | tr ',' '\n' | grep -c .)
python3 - "$QUEUES" "$direct" > /tmp/kueue-rendered.yaml <<'PY'
import re, sys
text, direct = open(sys.argv[1]).read(), int(sys.argv[2])
head, sep, rest = text.partition("metadata: {name: provers}")
m = re.search(r"\{name: nvidia.com/gpu, nominalQuota: (\d+)", rest) if sep else None
if m is None:
    sys.stdout.write(text)
    sys.exit(0)
gpus = max(0, int(m.group(1)) - direct)
sys.stdout.write(head + sep + rest[:m.start()] + "{name: nvidia.com/gpu, nominalQuota: %d" % gpus + rest[m.end():])
PY
echo "   provers GPU quota: $(grep -A20 'name: provers}' /tmp/kueue-rendered.yaml | grep -o 'nvidia.com/gpu, nominalQuota: [0-9]*' | head -1)"
run "${KUBECTL[@]}" apply -f /tmp/kueue-rendered.yaml
"${KUBECTL[@]}" -n gpu-operator get svc dcgm-exporter > /dev/null 2>&1 && echo "   dcgm-exporter: present" \
  || echo "   dcgm-exporter: MISSING (the GPU operator's install in step 2 failed: see sky ssh up's log); usage_report samples without GPU utilization"

step "4. SkyPilot config and the usage sampler"
if [ ! -f ~/.sky/config.yaml ]; then
  if [ "$mode" = --apply ]; then sed "s|infra: ssh/vy-nebius$|infra: ssh/$POOL|" "$SRC/sky-config.yaml" > ~/.sky/config.yaml; else echo "   would write ~/.sky/config.yaml (controller on ssh/$POOL)"; fi
  run sudo systemctl restart sky-api.service
elif ! sed "s|infra: ssh/vy-nebius$|infra: ssh/$POOL|" "$SRC/sky-config.yaml" | cmp -s - ~/.sky/config.yaml; then
  echo "   ~/.sky/config.yaml differs from sky-config.yaml: left as it is"
fi
# Jobs run as the image's user (uid 1000), not research: /workspace/jobs is where they write (runs, uv, the vLLM venv, caches).
run sudo install -d -m 1777 /workspace/jobs /workspace/jobs/cache
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

grep -q "metadata: {name: deployments-cpu}" "$QUEUES" && run sudo install -m 755 "$SRC/quiet_hour.sh" /usr/local/bin/vy-quiet-hour
grep -q "metadata: {name: deployments-cpu}" "$QUEUES" && for what in hold:12:30 release:13:30; do
  verb=${what%%:*}
  at=${what#*:}
  if [ ! -f "/etc/systemd/system/vy-quiet-$verb.timer" ]; then
    if [ "$mode" = --apply ]; then
      printf '[Unit]\nDescription=quiet hour: %s admissions to the vLLM deployments queues\n\n[Service]\nType=oneshot\nExecStart=/usr/local/bin/vy-quiet-hour %s\n' "$verb" "$verb" \
        | sudo tee "/etc/systemd/system/vy-quiet-$verb.service" > /dev/null
      printf '[Unit]\nDescription=quiet hour %s at %sZ\n\n[Timer]\nOnCalendar=*-*-* %s:00 UTC\n\n[Install]\nWantedBy=timers.target\n' "$verb" "$at" "$at" \
        | sudo tee "/etc/systemd/system/vy-quiet-$verb.timer" > /dev/null
    else
      echo "   would write vy-quiet-$verb.timer ($at UTC daily)"
    fi
  fi
done
run sudo systemctl daemon-reload
grep -q "metadata: {name: deployments-cpu}" "$QUEUES" && run sudo systemctl enable --now vy-quiet-hold.timer vy-quiet-release.timer
# the disk guard: every minute, hold every queue's admissions while /workspace is 80% full, release its own holds under 75%
run sudo install -m 755 "$SRC/disk_guard.sh" /usr/local/bin/vy-disk-guard
if [ ! -f /etc/systemd/system/vy-disk-guard.timer ]; then
  if [ "$mode" = --apply ]; then
    printf '[Unit]\nDescription=disk guard: hold Kueue admissions while /workspace is 80%% full\n\n[Service]\nType=oneshot\nExecStart=/usr/local/bin/vy-disk-guard check\n' \
      | sudo tee /etc/systemd/system/vy-disk-guard.service > /dev/null
    printf '[Unit]\nDescription=disk guard every minute\n\n[Timer]\nOnBootSec=1min\nOnUnitActiveSec=1min\n\n[Install]\nWantedBy=timers.target\n' \
      | sudo tee /etc/systemd/system/vy-disk-guard.timer > /dev/null
  else
    echo "   would write vy-disk-guard.service and vy-disk-guard.timer (every minute)"
  fi
fi
run sudo systemctl daemon-reload
run sudo systemctl enable --now vy-disk-guard.timer
# the store eviction: hourly, as ubuntu (the job store's owner), drop only local copies the remote store holds until 2500 GB are free
run sudo install -m 755 "$SRC/store_evict.sh" /usr/local/bin/vy-store-evict
# and the same for research's own store and check runs (/workspace/research/{store,runs}), as research, their owner
for u in store_evict.service store_evict.timer store_evict_research.service store_evict_research.timer; do
  run sudo install -m 644 "$SRC/$u" "/etc/systemd/system/vy-$(printf %s "$u" | sed 's/_/-/g')"
done
# research's pass also deletes the unused source trees of /workspace/research/src that are on origin (the file says how)
run sudo install -D -m 644 "$SRC/store_evict_src.conf" /etc/systemd/system/vy-store-evict-research.service.d/src.conf
run sudo systemctl daemon-reload
run sudo systemctl enable --now vy-store-evict.timer vy-store-evict-research.timer

step "5. the host's CUDA driver library for GPU-less pods (/workspace/jobs/cuda-driver)"
# A coverage cell's Build task has no GPU, and so no driver libraries in its pod, but importing vLLM's CUDA platform (#536) loads
# libcuda.so.1. The task finds this copy on LD_LIBRARY_PATH (jobs/config-run.yaml); without the /dev/nvidia* devices it can't reach a
# GPU. Re-copied when the host's driver changes.
drv=$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 2> /dev/null || true)
if [ -z "$drv" ] || [ ! -r "$drv" ]; then
  echo "   no libcuda.so.1 on this host: GPU-less Build tasks will fail to import vLLM's CUDA platform"
elif cmp -s "$drv" "/workspace/jobs/cuda-driver/$(basename "$drv")"; then
  echo "   /workspace/jobs/cuda-driver has $(basename "$drv")"
else
  run install -d -m 755 /workspace/jobs/cuda-driver
  run cp -p "$drv" /workspace/jobs/cuda-driver/
  run ln -sfn "$(basename "$drv")" /workspace/jobs/cuda-driver/libcuda.so.1
fi

step "done"
echo "   check:  sky check ssh && sky gpus list --infra ssh/$POOL && ${KUBECTL[*]} get clusterqueues"
echo "   client: ssh -N -L 46580:127.0.0.1:46580 $USER@<public ip>  then  sky api login -e http://127.0.0.1:46580"
