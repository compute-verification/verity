#!/usr/bin/env bash
# Submit one job to the Nebius cluster's Kueue queues, from any lane checkout:
#
#   tools/research/src/research/pods/nebius/sky/submit.sh TEMPLATE NAME [sky jobs launch flags...]
#   e.g.  .../submit.sh prover-dev fri-sweep-1 --env CMD='cargo test --release -p flock'
#         .../submit.sh config-run row-23 --env ROW=... --env ROLE=B1 --env REPO=Qwen/Qwen2.5-1.5B --env REVISION=<sha>
#
# TEMPLATE is a file in jobs/ (config-run, port-capture, prover-dev, prover-bench). It:
#   1. installs the SkyPilot 0.13.0 client if missing, and opens the ssh tunnel to the API server on vy-nebius-1
#      (127.0.0.1:46580, through the research key) unless one answers;
#   2. ships this checkout's git-visible files to /workspace/research/trees/$VY_LANE on the host (`research pods sync`):
#      jobs run from there (SRC), not from an upload;
#   3. runs `sky jobs launch` with the R2 credentials as secrets (from this environment) and prints the job's queue line.
# VY_ROW_CLASS=small|dense|long|tp2 sizes config-run-split's two tasks (its header's table).
# VY_LANE defaults to this checkout's branch name. KUBECONFIG is unset: the SkyPilot client forwards it to the server.
# Another server's cluster (its own API server): VY_MACHINE=vy-nebius-2 VY_POOL=vy-nebius-2 VY_SKY_PORT=46581.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,15p' "$0" >&2; exit 2; }
template=$1 name=$2
shift 2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
yaml=$here/jobs/$template.yaml
[ -f "$yaml" ] || { echo "no template $template (jobs/: $(cd "$here/jobs" && ls | sed 's/.yaml//' | tr '\n' ' '))" >&2; exit 2; }
machine=${VY_MACHINE:-vy-nebius-1}
lane=${VY_LANE:-$(git -C "$repo" rev-parse --abbrev-ref HEAD | tr '/' '-')}
pool=${VY_POOL:-vy-nebius}
port=${VY_SKY_PORT:-46580}
endpoint=http://127.0.0.1:$port
unset KUBECONFIG
export PATH=$HOME/.local/bin:$PATH
research() { PYTHONPATH=$repo/tools/research/src python3 -m research "$@"; }

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
if [ -n "${VY_ROW_CLASS:-}" ]; then   # config-run-split: the build and gpu tasks' memory (and TP2's GPUs) per row class
  rendered=$(mktemp --suffix=.yaml)
  python3 - "$yaml" "$VY_ROW_CLASS" > "$rendered" <<'PY'
import re, sys
text, cls = open(sys.argv[1]).read(), sys.argv[2]
table = {"small": (64, 96, 1), "dense": (160, 192, 1), "long": (160, 256, 1), "tp2": (160, 384, 2)}
if cls not in table:
    sys.exit(f"VY_ROW_CLASS={cls}: one of {sorted(table)}")
build, gpu, n = table[cls]
parts = text.split("\n---\n")
if len(parts) != 3:
    sys.exit("VY_ROW_CLASS applies to a two-task pipeline template (config-run-split)")
parts[1] = re.sub(r"(?m)^  memory: \d+$", f"  memory: {build}", parts[1])
parts[2] = re.sub(r"(?m)^  memory: \d+$", f"  memory: {gpu}", parts[2])
parts[2] = re.sub(r"(?m)^  accelerators: RTXPRO6000:\d+$", f"  accelerators: RTXPRO6000:{n}", parts[2])
sys.stdout.write("\n---\n".join(parts))
PY
  yaml=$rendered
fi
tunnel    # the sync can take minutes, long enough for a tunnel to drop
sky jobs launch -y -d -n "$name" "$yaml" --infra "ssh/$pool" --env "SRC=$dest" "${secrets[@]}" "$@"
sky jobs queue 2>/dev/null | awk -v n="$name" 'NR <= 3 || $0 ~ n' | head -6
