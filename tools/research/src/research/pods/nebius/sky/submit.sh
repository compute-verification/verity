#!/usr/bin/env bash
# Submit one job to the Nebius cluster's Kueue queues, from any lane checkout:
#
#   tools/research/src/research/pods/nebius/sky/submit.sh TEMPLATE NAME [sky jobs launch flags...]
#   e.g.  .../submit.sh prover-dev fri-sweep-1 --env CMD='cargo test --release -p flock'
#         .../submit.sh config-run row-23 --env ROW=... --env ROLE=B1 --env REPO=Qwen/Qwen2.5-1.5B --env REVISION=<sha>
#
# It needs uv, rsync, ssh, git, python3 and curl on this machine (it checks, and names what's missing). It refuses (exit 3) unless
# this checkout's sky/ (the templates and this script) equals origin/infra/nebius, and names the commit to merge; --allow-stale
# submits anyway, loudly.
# TEMPLATE is a file in jobs/ (config-run, config-run-row, port-capture, prover-dev, prover-bench). It:
#   1. installs the SkyPilot 0.13.0 client if missing, and opens the ssh tunnel to the API server on vy-nebius-1
#      (127.0.0.1:46580, through the research key) unless one answers;
#   2. ships this checkout's git-visible files to /workspace/research/trees/$VY_LANE on the host (`research pods sync`):
#      jobs run from there (SRC), not from an upload;
#   3. runs `sky jobs launch` with the R2 credentials as secrets (from this environment) and prints the job's queue line.
# config-run's two tasks: VY_ROW_CLASS=small|dense|long|moe|b64|tp2 (its header's table), or VY_BUILD_MEMORY / VY_GPU_MEMORY
# in GB per cell, which win over the class.
# VY_LANE defaults to this checkout's branch name. KUBECONFIG is unset: the SkyPilot client forwards it to the server.
# Another server's cluster (its own API server): VY_MACHINE=vy-nebius-2 VY_POOL=vy-nebius-2 VY_SKY_PORT=46581.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,15p' "$0" >&2; exit 2; }
template=$1 name=$2
shift 2
allow_stale=0
args=()
for a in "$@"; do if [ "$a" = --allow-stale ]; then allow_stale=1; else args+=("$a"); fi; done
set -- ${args[@]+"${args[@]}"}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
yaml=$here/jobs/$template.yaml
[ -f "$yaml" ] || { echo "no template $template (jobs/: $(cd "$here/jobs" && ls | sed 's/.yaml//' | tr '\n' ' '))" >&2; exit 2; }
# The templates and this script must be the shared ones: a job submitted from a stale checkout ran old templates (port-capture job
# 35, 36 minutes lost). Refused unless this checkout's sky/ equals $VY_INFRA_REMOTE's infra/nebius, or --allow-stale.
remote=${VY_INFRA_REMOTE:-origin}
ref=$remote/${VY_INFRA_BRANCH:-infra/nebius}
rel=$(git -C "$here" rev-parse --show-prefix); rel=${rel%/}
stale=""
if ! GIT_TERMINAL_PROMPT=0 git -C "$repo" fetch -q "$remote" "${ref#"$remote"/}" 2> /dev/null < /dev/null; then   # never prompt
  stale="could not fetch $ref, so this checkout's templates can't be checked"
elif ! git -C "$repo" cat-file -e "$ref:$rel/jobs/$template.yaml" 2> /dev/null; then
  stale="jobs/$template.yaml is not on $ref (a private copy)"
elif ! git -C "$repo" diff --quiet "$ref" -- "$rel"; then
  stale="$rel differs from $ref ($(git -C "$repo" diff --name-only "$ref" -- "$rel" | sed "s|^$rel/||" | tr '\n' ' '))"
fi
if [ -n "$stale" ]; then
  echo "submit.sh: STALE TEMPLATES: $stale." >&2
  echo "  $ref is $(git -C "$repo" log -1 --format='%h %s' "$ref" 2> /dev/null || echo '?'): git fetch $remote && git merge $ref" >&2
  if [ "$allow_stale" = 1 ]; then
    echo "  --allow-stale: submitting anyway" >&2
  else
    echo "  refused (pass --allow-stale to submit anyway)" >&2
    exit 3
  fi
fi
machine=${VY_MACHINE:-vy-nebius-1}
lane=${VY_LANE:-$(git -C "$repo" rev-parse --abbrev-ref HEAD | tr '/' '-')}
pool=${VY_POOL:-vy-nebius}
port=${VY_SKY_PORT:-46580}
endpoint=http://127.0.0.1:$port
unset KUBECONFIG
export PATH=$HOME/.local/bin:$PATH
research() { PYTHONPATH=$repo/tools/research/src python3 -m research "$@"; }

missing=()
for tool in uv rsync ssh git python3 curl; do command -v "$tool" > /dev/null || missing+=("$tool"); done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "submit.sh: missing on this machine: ${missing[*]}. Install them first, e.g." >&2
  echo "  sudo apt-get install -y rsync openssh-client git python3 curl   # and uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 2
fi
tunnel() {
  # the ssh tunnel to the API server, (re)opened unless it answers. A dropped tunnel whose ssh still holds the port is closed
  # first (ExitOnForwardFailure would refuse the new one), and keepalives end a dead connection within ~45 s
  curl -sf "$endpoint/api/health" > /dev/null && return 0
  pkill -f "^(/usr/bin/)?ssh .*-L $port:127.0.0.1:46580 " 2> /dev/null || true
  local ssh_cmd
  ssh_cmd=$(research pods ssh "$machine" --print)
  # shellcheck disable=SC2086
  ${ssh_cmd% *} -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -L "$port:127.0.0.1:46580" "${ssh_cmd##* }"
  for _ in $(seq 20); do curl -sf "$endpoint/api/health" > /dev/null && return 0; sleep 1; done
  echo "submit.sh: the API server doesn't answer at $endpoint through the tunnel to $machine" >&2
  return 1
}

command -v sky > /dev/null || uv tool install -q "skypilot[kubernetes]==0.13.0"
tunnel
sky api info 2>/dev/null | grep -q "$endpoint" || sky api login -e "$endpoint" > /dev/null
dest=/workspace/research/trees/$lane
research pods sync "$machine" "$repo" --dest "$dest"
secrets=()
for s in R2_ENDPOINT R2_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do secrets+=(--secret "$s"); done
if [ -n "${VY_ROW_CLASS:-}${VY_BUILD_MEMORY:-}${VY_GPU_MEMORY:-}${VY_REPLAY_MEMORY:-}" ]; then   # config-run: its tasks' memory (and TP2's GPUs)
  rendered=$(mktemp --suffix=.yaml)
  python3 - "$yaml" "${VY_ROW_CLASS:-}" "${VY_BUILD_MEMORY:-}" "${VY_GPU_MEMORY:-}" "${VY_REPLAY_MEMORY:-}" > "$rendered" <<'PY'
import re, sys
text, cls, build_gb, gpu_gb, replay_gb = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
table = {"small": (64, 64, 1), "dense": (160, 170, 1), "long": (160, 170, 1), "moe": (256, 170, 1), "b64": (256, 170, 1),
         "tp2": (160, 340, 2)}                          # the gpu task: batch 8+ Commits use ~130 GiB (epoch-run 16:35Z)
if cls and cls not in table:
    sys.exit(f"VY_ROW_CLASS={cls}: one of {sorted(table)}")
parts = text.split("\n---\n")
if len(parts) not in (3, 4):
    sys.exit("VY_ROW_CLASS / VY_BUILD_MEMORY / VY_GPU_MEMORY / VY_REPLAY_MEMORY apply to a pipeline template (config-run)")
build, gpu, n = table.get(cls) or (None, None, None)
build, gpu = int(build_gb or build or 0), int(gpu_gb or gpu or 0)
if build:
    parts[1] = re.sub(r"(?m)^  memory: \d+", f"  memory: {build}", parts[1])
    parts[1] = re.sub(r'(?m)^  BUILD_RAM_BUDGET_GB: "\d+"$', f'  BUILD_RAM_BUDGET_GB: "{build}"', parts[1])
if gpu:
    parts[2] = re.sub(r"(?m)^  memory: \d+", f"  memory: {gpu}", parts[2])
if n:
    parts[2] = re.sub(r"(?m)^  accelerators: RTXPRO6000:\d+$", f"  accelerators: RTXPRO6000:{n}", parts[2])
    parts[2] = re.sub(r"(?m)^  cpus: \d+", f"  cpus: {4 * n}", parts[2])                           # the GPU queue's 4 vCPU per GPU
if replay_gb and len(parts) == 4:
    parts[3] = re.sub(r"(?m)^  memory: \d+", f"  memory: {int(replay_gb)}", parts[3])
sys.stdout.write("\n---\n".join(parts))
PY
  yaml=$rendered
fi
# At most VY_MAX_WAITING_CELLS vLLM deployments' tasks (default 4) and VY_MAX_WAITING jobs in all (default 6) wait in Kueue: a job waiting for
# quota still holds one of the SkyPilot controller's 8 launch slots (provision_timeout: -1), and with every slot held no other job
# reaches Kueue. Only SkyPilot's own pods count: the dispatcher's, the lease holders' and commit-pack's Kueue Jobs hold no launch
# slot (they refused every Sky submission for 2 h, 4:10 PM PDT Sep 30). A queue it can't read doesn't block: this caps throughput
# loss, it guards no spend.
wl=$(mktemp)
research pods ssh "$machine" -- sudo k3s kubectl get workloads -A -o json > "$wl" 2> /dev/null || true
python3 - "$wl" "$template" "${VY_MAX_WAITING_CELLS:-4}" "${VY_MAX_WAITING:-6}" <<'WAITING' || { rm -f "$wl"; exit 4; }
import json, sys
path, template, cells_max, all_max = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
try:
    items = json.load(open(path)).get("items", [])
except (OSError, ValueError, AttributeError):
    sys.exit(0)
def waiting(w):
    st = w.get("status") or {}
    done = any(c.get("type") == "Finished" and c.get("status") == "True" for c in st.get("conditions") or [])
    return not st.get("admission") and not done
def sky(w):   # a SkyPilot pod's workload; a batch/v1 Job's (dispatcher, holders, commit-pack) holds no launch slot
    m = w.get("metadata") or {}
    return any(o.get("kind") == "Pod" for o in m.get("ownerReferences") or []) or str(m.get("name", "")).startswith("pod-")
pending = [w for w in items if waiting(w) and sky(w)]
cells = sum((w.get("spec") or {}).get("queueName") in ("deployments-cpu", "deployments-gpu", "circuits") for w in pending)
why = ""
if template in ("config-run", "config-run-row") and cells >= cells_max:
    why = f"{cells} vLLM deployments already wait in Kueue (at most {cells_max}, VY_MAX_WAITING_CELLS): submit when one is admitted"
elif len(pending) >= all_max:
    why = f"{len(pending)} jobs already wait in Kueue (at most {all_max} of SkyPilot's 8 launch slots, VY_MAX_WAITING)"
if why:
    print(f"submit.sh: {why}", file=sys.stderr)
    sys.exit(4)
WAITING
rm -f "$wl"
tunnel    # the sync can take minutes, long enough for a tunnel to drop
sky jobs launch -y -d -n "$name" "$yaml" --infra "ssh/$pool" --env "SRC=$dest" "${secrets[@]}" "$@"
sky jobs queue 2>/dev/null | awk -v n="$name" 'NR <= 3 || $0 ~ n' | head -6
