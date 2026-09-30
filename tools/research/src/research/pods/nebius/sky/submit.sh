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
# in GB per cell (e.g. `verity-vllm sweep plan`'s ram_gb), which win over the class.
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
if ! git -C "$repo" fetch -q "$remote" "${ref#"$remote"/}" 2> /dev/null; then
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
if [ -n "${VY_ROW_CLASS:-}${VY_BUILD_MEMORY:-}${VY_GPU_MEMORY:-}" ]; then   # config-run: its two tasks' memory (and TP2's GPUs)
  rendered=$(mktemp --suffix=.yaml)
  python3 - "$yaml" "${VY_ROW_CLASS:-}" "${VY_BUILD_MEMORY:-}" "${VY_GPU_MEMORY:-}" > "$rendered" <<'PY'
import re, sys
text, cls, build_gb, gpu_gb = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3], sys.argv[4]
table = {"small": (64, 64, 1), "dense": (160, 192, 1), "long": (160, 256, 1), "moe": (256, 384, 1), "b64": (256, 512, 1),
         "tp2": (160, 384, 2)}
if cls and cls not in table:
    sys.exit(f"VY_ROW_CLASS={cls}: one of {sorted(table)}")
parts = text.split("\n---\n")
if len(parts) != 3:
    sys.exit("VY_ROW_CLASS / VY_BUILD_MEMORY / VY_GPU_MEMORY apply to a two-task pipeline template (config-run)")
build, gpu, n = table.get(cls) or (None, None, None)
build, gpu = int(build_gb or build or 0), int(gpu_gb or gpu or 0)
if build:
    parts[1] = re.sub(r"(?m)^  memory: \d+$", f"  memory: {build}", parts[1])
    parts[1] = re.sub(r'(?m)^  BUILD_RAM_BUDGET_GB: "\d+"$', f'  BUILD_RAM_BUDGET_GB: "{build}"', parts[1])
if gpu:
    parts[2] = re.sub(r"(?m)^  memory: \d+$", f"  memory: {gpu}", parts[2])
if n:
    parts[2] = re.sub(r"(?m)^  accelerators: RTXPRO6000:\d+$", f"  accelerators: RTXPRO6000:{n}", parts[2])
sys.stdout.write("\n---\n".join(parts))
PY
  yaml=$rendered
fi
tunnel    # the sync can take minutes, long enough for a tunnel to drop
sky jobs launch -y -d -n "$name" "$yaml" --infra "ssh/$pool" --env "SRC=$dest" "${secrets[@]}" "$@"
sky jobs queue 2>/dev/null | awk -v n="$name" 'NR <= 3 || $0 ~ n' | head -6
