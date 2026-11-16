#!/usr/bin/env bash
# Publish the Kueue jobs' attempts that never reached the evidence store. Run on the server (as research, which sudoes to the job
# user); the R2 credentials come on stdin, four lines (R2_ENDPOINT, R2_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), never argv:
#
#   printf '%s\n' "$R2_ENDPOINT" "$R2_BUCKET" "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" | \
#     research pods ssh vy-nebius-1 -- bash /workspace/research/checkout/tools/research/src/research/pods/nebius/sky/backfill_attempts.sh [--dry-run]
#
# A run under /workspace/jobs/runs whose pod is gone (its events' span names the pod) and whose sampler has been silent for 10 min
# was killed mid-run (preempted, or its job cancelled): its status becomes `cancelled` (reason=preempted-pod-gone), then every
# terminal run the store's remote lacks is published (`research data attempt publish`, which skips what's already there).
set -euo pipefail
dry=0
[ "${1:-}" = --dry-run ] && dry=1
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
research_src=$(cd "$here/../../../.." && pwd)
runs=/workspace/jobs/runs
creds=$(mktemp)
chmod 600 "$creds"
trap 'sudo -n rm -f "$creds" 2> /dev/null || rm -f "$creds"' EXIT
head -n 4 > "$creds"
[ "$(grep -c . "$creds")" = 4 ] || { echo "backfill: want the 4 credential lines on stdin" >&2; exit 2; }
sudo chown 1000 "$creds"
live=$(KUBECONFIG=$HOME/.kube/config kubectl get pods -n default -o name | sed 's|pod/||')
now=$(date +%s)
orphans=()
for d in "$runs"/r*; do
  [ -f "$d/status.json" ] || continue
  state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state"))' "$d/status.json")
  [ "$state" = running ] || continue
  pod=$(grep -o 'run#1@[^:]*' "$d/events.jsonl" 2> /dev/null | head -1 | cut -d@ -f2)
  idle=$(( (now - $(stat -c %Y "$d/resources.jsonl" 2> /dev/null || echo 0)) / 60 ))
  if [ -n "$pod" ] && ! grep -qx "$pod" <<< "$live" && [ "$idle" -ge 10 ]; then orphans+=("$(basename "$d")"); fi
done
echo "backfill: ${#orphans[@]} orphaned run(s): ${orphans[*]:-none}"
[ "$dry" = 1 ] && exit 0
# shellcheck disable=SC2016
sudo -u "#1000" -H env RUNS="$runs" CREDS="$creds" SRC="$research_src" ORPHANS="${orphans[*]:-}" bash -c '
  set -u
  { read -r R2_ENDPOINT; read -r R2_BUCKET; read -r AWS_ACCESS_KEY_ID; read -r AWS_SECRET_ACCESS_KEY; } < "$CREDS"
  export R2_ENDPOINT R2_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
  export RESEARCH_STORE=/workspace/jobs/store UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python
  export PYTHONPATH=$SRC
  py=/workspace/jobs/bin/py312
  cd "$RUNS"
  for r in $ORPHANS; do
    $py -m research tele status --attempt-dir "$r" cancelled reason=preempted-pod-gone recorded_by=backfill > /dev/null
    $py -m research tele classify --attempt-dir "$r" --signal SIGKILL > /dev/null 2>&1 || true
  done
  $py -m research data attempt publish --all --runs-dir "$RUNS" 2>&1 | tail -3
'
