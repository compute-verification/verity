#!/usr/bin/env bash
# [E6 CURVE 2026-09-13, D55] Laptop side: create the ONE CPU pod `cv-e6-repin` (cpu5g 16 vCPU / 64 GB, $0.74/h; volume h5uk3omqsl at /workspace),
# ship the guard (pod_guard.sh + runpod.py + ~/.runpod/config.toml, as deadman.sh does), a `git archive` of THIS tree, a uv python 3.12 venv, then
# start `cv_repin_chain.sh` DETACHED (setsid nohup) and return.  After this returns nothing depends on the laptop: the chain writes to the volume,
# guard_trap/guard_finish terminate the pod when it ends, guard_start terminates it after 20 idle minutes, the controller `cv-int-deadman`
# terminates it at 4 h of age / the cap / the deadline.
#
#   scripts/cv_repin_launch.sh --geometry '{"kernel":"fa2-splitkv","kBlockM":64,"kBlockN":128,"num_splits":1}' \
#                                       --geometry-source 'E10 out/curve/e10/KERNEL-GEOMETRY.md §A (16:18Z, MEASURED on the 4090): flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 64, 128, 4>>, num_splits 1, all 720 launches at P in {63,64,65,127,128,129} x 4 tokens'
#   LEGACY_ONLY=1 scripts/cv_repin_launch.sh   # no geometry: only the legacy regression builds (any tree)
#   POD_ID=... scripts/cv_repin_launch.sh ...  # reuse an existing pod (skips create)
set -euo pipefail
GEOMETRY=""; GEOMETRY_SOURCE=""
while [ $# -gt 0 ]; do case "$1" in
  --geometry) GEOMETRY=$2; shift 2;; --geometry-source) GEOMETRY_SOURCE=$2; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
PY=${PY:-/Users/danielreuter/projects/veritor/.venv/bin/python}
RP="$PY zk/campaign/runpod.py"
NAME=cv-e6-repin; VOL=h5uk3omqsl; DC=${DC:-EU-RO-1}
PODS=out/scale/e6/pods; mkdir -p "$PODS"
HASH=$(git rev-parse --short=9 HEAD)
if [ -z "${LEGACY_ONLY:-}" ]; then
  [ -n "$GEOMETRY" ] && [ -n "$GEOMETRY_SOURCE" ] || { echo "REFUSED: --geometry and --geometry-source are required (D57: the pin records the measured value AND its provenance); LEGACY_ONLY=1 for the regression builds alone"; exit 2; }
  $PY -c "import json,sys; g=json.loads(sys.argv[1]); assert all(k in g for k in ('kernel','kBlockM','kBlockN','num_splits')), g" "$GEOMETRY"
  if ! $PY -c "import sys; sys.path[:0]=['src','.']; from verity_vllm.registry import b1; sys.exit(0 if 'Serve_v4' in getattr(b1,'SERVE_FAMILIES',{}) else 1)"; then
    echo "REFUSED: this tree ($HASH) has no b1.Serve_v4 / serve_program (E7's D57 static e65daae52 not merged) -- geometry pins cannot be built; LEGACY_ONLY=1 runs the regression builds"; exit 3
  fi
fi
echo "price row: cpu5g 16 vCPU / 64 GB RAM = \$0.74/h (integrator's line, CURVE brief); budget <= \$1.5 -> <= 2 h; expected ~1 h"
if [ -z "${POD_ID:-}" ]; then
  # flavor/vcpu overridable when cpu5g 16 is out of stock in the volume's DC (cpu5m/cpu3m = 8 GB per vCPU: --vcpu 8 is the same 64 GB box)
  $RP create --name $NAME --cpu "${CPU_FLAVOR:-cpu5g}" --vcpu "${VCPU:-16}" --disk 40 --network-volume $VOL --volume-mount /workspace --data-center "$DC" > "$PODS/$NAME.create.json"
  POD_ID=$($PY -c "import json; print(json.load(open('$PODS/$NAME.create.json'))['id'])")
fi
echo "$POD_ID" > "$PODS/$NAME.id"; echo "pod $POD_ID ($NAME) created $(date -u +%FT%TZ) tree $HASH" | tee -a "$PODS/$NAME.log"
SSH_LINE=$($RP wait-ssh "$POD_ID" | tail -1)
PORT=$(printf '%s\n' "$SSH_LINE" | sed -E 's/.* -p ([0-9]+) .*/\1/'); HOST=$(printf '%s\n' "$SSH_LINE" | sed -E 's/.*root@([^ ]+).*/\1/')
SSHOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -i "$HOME/.runpod/ssh/runpodctl-ssh-key")
SSH=(ssh "${SSHOPTS[@]}" -p "$PORT" "root@$HOST"); SCP=(scp "${SSHOPTS[@]}" -P "$PORT")
printf '#!/usr/bin/env bash\nexec ssh %s -p %s root@%s "$@"\n' "${SSHOPTS[*]}" "$PORT" "$HOST" > "$PODS/$NAME.ssh"; chmod +x "$PODS/$NAME.ssh"
echo "ssh: $PODS/$NAME.ssh"

# guard + credentials (exactly as deadman.sh ships them)
"${SSH[@]}" 'mkdir -p /root/.runpod /root/dm /workspace/cv/e6'
"${SCP[@]}" zk/campaign/pod_guard.sh zk/campaign/runpod.py "root@$HOST:/root/dm/"
"${SCP[@]}" "$HOME/.runpod/config.toml" "root@$HOST:/root/.runpod/config.toml"
"${SSH[@]}" "echo $POD_ID > /root/dm/POD_ID; mountpoint -q /workspace && df -h /workspace | tail -1 || { echo 'REFUSED: /workspace is not the volume'; exit 4; }"

# the tree (git archive of HEAD; the pods hold an archive, not a repository) + venv
git archive --format=tar HEAD | "${SSH[@]}" 'rm -rf /workspace/cv/e6/tree && mkdir -p /workspace/cv/e6/tree && tar -x -C /workspace/cv/e6/tree'
"${SSH[@]}" "echo $HASH > /workspace/cv/e6/TREE_HASH; cd /workspace/cv/e6 && export PATH=\$HOME/.local/bin:\$PATH && (command -v uv >/dev/null || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null) && \
  ([ -x venv/bin/python ] || uv venv --python 3.12 venv >/dev/null) && uv pip install --python venv/bin/python 'numpy>=2.3' 'networkx>=3.6' safetensors pytest >/dev/null && venv/bin/python -c 'import numpy, networkx, safetensors; print(\"py deps ok\", numpy.__version__)' && \
  cp tree/scripts/cv_repin_chain.sh . && chmod +x cv_repin_chain.sh && nproc && free -g | head -2"

# the measured geometry + provenance the chain pins (absent under LEGACY_ONLY: the chain refuses the geometry pins by name)
if [ -n "$GEOMETRY" ]; then
  $PY -c "import json,sys; print(json.dumps({'attention': json.loads(sys.argv[1]), 'source': sys.argv[2], 'written_by': 'cv_repin_launch.sh', 'tree': sys.argv[3]}, indent=1))" "$GEOMETRY" "$GEOMETRY_SOURCE" "$HASH" \
    | "${SSH[@]}" 'cat > /workspace/cv/e6/GEOMETRY.json; cat /workspace/cv/e6/GEOMETRY.json'
else
  "${SSH[@]}" 'rm -f /workspace/cv/e6/GEOMETRY.json'
fi

# start the chain DETACHED; from here on the laptop may die
"${SSH[@]}" 'cd /workspace/cv/e6 && (setsid nohup bash /workspace/cv/e6/cv_repin_chain.sh > /workspace/cv/e6/chain.log 2>&1 &) ; sleep 5; pgrep -af cv_repin_chain | head -3; tail -3 /workspace/cv/e6/chain.log; tail -2 /root/dm/guard.log'
echo "chain started $(date -u +%FT%TZ) on $POD_ID; follow: $PODS/$NAME.ssh 'tail -f /workspace/cv/e6/chain.log'" | tee -a "$PODS/$NAME.log"
