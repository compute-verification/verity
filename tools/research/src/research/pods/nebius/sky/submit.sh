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
# VY_LANE defaults to this checkout's branch name. KUBECONFIG is unset: the SkyPilot client forwards it to the server.
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
endpoint=http://127.0.0.1:46580
unset KUBECONFIG
export PATH=$HOME/.local/bin:$PATH
research() { PYTHONPATH=$repo/tools/research/src python3 -m research "$@"; }

command -v sky > /dev/null || uv tool install -q "skypilot[kubernetes]==0.13.0"
if ! curl -sf "$endpoint/api/health" > /dev/null; then
  ssh_cmd=$(research pods ssh "$machine" --print)
  # shellcheck disable=SC2086
  ${ssh_cmd% *} -f -N -o ExitOnForwardFailure=yes -L 46580:127.0.0.1:46580 "${ssh_cmd##* }"
  for _ in $(seq 20); do curl -sf "$endpoint/api/health" > /dev/null && break; sleep 1; done
fi
sky api info 2>/dev/null | grep -q "$endpoint" || sky api login -e "$endpoint" > /dev/null
dest=/workspace/research/trees/$lane
research pods sync "$machine" "$repo" --dest "$dest"
secrets=()
for s in R2_ENDPOINT R2_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do secrets+=(--secret "$s"); done
sky jobs launch -y -d -n "$name" "$yaml" --env "SRC=$dest" "${secrets[@]}" "$@"
sky jobs queue 2>/dev/null | awk -v n="$name" 'NR <= 3 || $0 ~ n' | head -6
