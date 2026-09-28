#!/usr/bin/env bash
# Node 2's custody: its stores (/workspace/jobs/store, /workspace/research/store) have no remote and the node holds no R2 credential,
# so every ended run under its run roots goes home as n2_commit.sh's home() sends a Commit: rsync to node 1, `research data attempt
# publish` into node 1's /workspace/jobs/store, then `research data push` from a one-off Job on node 1 with the Secret research-r2
# (the credential never leaves Kubernetes). n2_commit.sh and n2_build.sh name this as their runs' custody (RESEARCH_LOCAL_ONLY),
# without which `research run` refuses to start on a store with no remote.
#
#   on node 2:  n2_custody.sh [--dry-run]   one round (the user timer n2-custody.timer, every 10 min); --dry-run lists what it would push
#
# A run has ended when its status says so or its process is gone: a runner killed before it recorded an end (an OOM kill of its
# scope) is published with --force and keeps the state it last wrote. A run its own custody preserved (.custody "preserved": true)
# is only marked; any other waits VY_CUSTODY_SETTLE_S after its records last changed, so a runner's own custody finishes first.
# A run is marked in STATE/pushed/ once node 1's Job completes. A failure is logged and retried every round, never given up, and
# the round exits 1, which systemd records (`systemctl --user status n2-custody`).
set -u
ROOTS=${VY_CUSTODY_ROOTS:-/workspace/jobs/runs=/workspace/jobs/runs,/workspace/research/runs=/workspace/jobs/runs-n2}
STATE=${VY_CUSTODY_STATE:-/workspace/research/custody-push}
SETTLE_S=${VY_CUSTODY_SETTLE_S:-900}
N1=${VY_N1:-vy-n1}
DRY=0
[ "${1:-}" = --dry-run ] && DRY=1
KUBE="KUBECONFIG=\$HOME/.kube/config kubectl"
RESEARCH_N1="cd /workspace/jobs && sudo -n -u '#1000' env UV_NO_CONFIG=1 RESEARCH_STORE=/workspace/jobs/store \
  UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python"
mkdir -p "$STATE/pushed" "$STATE/failures"
exec 9> "$STATE/.lock"
flock -n 9 || { echo "n2_custody.sh: another round holds $STATE/.lock"; exit 0; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATE/log"; }

# "CAMPAIGN STATE AGE_S": STATE is the status's, or dead for a run still marked running whose process is gone; AGE_S since its
# records last changed (status.json, .custody)
info() {
  python3 - "$1" << 'EOF'
import json, os, re, sys, time
d = sys.argv[1]
try:
    m = re.search(r'"campaign": ?"([^"]*)"', open(f"{d}/job.json").read())
except OSError:
    m = None
try:
    st = json.load(open(f"{d}/status.json"))
except (OSError, ValueError):
    st = {}
state = st.get("state") or "running"
if state == "running":
    t = (st.get("transitions") or [{}])[-1]
    try:
        alive = int(open(f"/proc/{t['pid']}/stat").read().rsplit(")", 1)[1].split()[19]) == t["starttime_ticks"]
    except (KeyError, OSError, ValueError, IndexError):
        alive = not st          # no status yet: a run that is starting
    state = "running" if alive else "dead"
mt = max([os.path.getmtime(p) for p in (f"{d}/status.json", f"{d}/.custody") if os.path.exists(p)] or [time.time()])
print(m.group(1) if m else "-", state, int(time.time() - mt))
EOF
}

# the research tree node 1 publishes and pushes with: VY_CUSTODY_SRC, else node 1's newest complete tree that `research run --on`
# shipped (an unfinished one is <sha>.partial-*)
n1_src() {
  if [ -n "${VY_CUSTODY_SRC:-}" ]; then echo "$VY_CUSTODY_SRC"; return; fi
  ssh -n "$N1" 'for t in $(ls -td /workspace/research/src/*/ | grep -E "/[0-9a-f]{40}/$"); do
    [ -f "${t}tools/research/src/research/store/attempt.py" ] && { echo "${t%/}"; break; }; done'
}

push_run() {  # push_run ROOT DST RUN_ID STATE
  local root=$1 dst=$2 r=$3 name force=
  [ "$4" = dead ] && force=--force
  rsync -a --rsync-path="sudo -n rsync" --chown=1000:1000 --mkpath "$root/$r/" "$N1:$dst/$r/" || return 1
  ssh -n "$N1" "$RESEARCH_N1 RESEARCH_RUNS=$dst PYTHONPATH=$SRC/tools/research/src /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data attempt publish $force $dst/$r" || return 1
  name=n2-custody-$(echo "$r" | tr -cd 'a-z0-9-' | tail -c 40)
  ssh "$N1" "$KUBE delete job $name --ignore-not-found > /dev/null && $KUBE create -f - > /dev/null \
    && $KUBE wait --for=condition=complete --timeout=1800s job/$name > /dev/null" << EOF || return 1
{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "$name", "labels": {"verity.dev/lane": "n2-custody", "verity.dev/kind": "custody"}},
 "spec": {"backoffLimit": 2, "ttlSecondsAfterFinished": 3600, "template": {"metadata": {"labels": {"verity.dev/lane": "n2-custody"}}, "spec": {
  "restartPolicy": "Never", "nodeSelector": {"verity.dev/server": "vy-nebius-1"},
  "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
  "securityContext": {"runAsUser": 1000, "runAsGroup": 1000},
  "containers": [{"name": "main", "image": "${VY_DISPATCH_IMAGE:-us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216}",
   "imagePullPolicy": "IfNotPresent",
   "command": ["taskset", "-c", "96-127", "bash", "-c", "cd /workspace/jobs && HOME=/tmp UV_NO_CONFIG=1 PYTHONPATH=$SRC/tools/research/src RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=$dst UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data push $r"],
   "envFrom": [{"secretRef": {"name": "research-r2"}}],
   "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
   "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}]}],
  "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}}]}}}}
EOF
}

SRC=
failed=0
IFS=, read -ra pairs <<< "$ROOTS"
for pair in "${pairs[@]}"; do
  root=${pair%%=*} dst=${pair#*=}
  for d in "$root"/r*/; do
    [ -d "$d" ] || continue
    r=$(basename "$d")
    [ -e "$STATE/pushed/$r" ] && continue
    read -r camp state age < <(info "$d")
    [ "$state" = running ] && continue
    if grep -qs '"preserved": true' "$d/.custody"; then
      [ "$DRY" = 1 ] || touch "$STATE/pushed/$r"
      continue
    fi
    [ "$age" -lt "$SETTLE_S" ] && continue
    if [ "$DRY" = 1 ]; then
      echo "would push $root/$r ($camp, $state, $(du -sh "$d" | cut -f1))"
      continue
    fi
    if [ -z "$SRC" ]; then
      SRC=$(n1_src)
      [ -n "$SRC" ] || { log "no research tree on $N1 to publish with (VY_CUSTODY_SRC, or /workspace/research/src/<sha>)"; exit 1; }
    fi
    if push_run "$root" "$dst" "$r" "$state"; then
      touch "$STATE/pushed/$r"
      rm -f "$STATE/failures/$r"
      log "$root/$r ($camp, $state, $(du -sh "$d" | cut -f1)): on node 1 and pushed to R2 (research from ${SRC##*/})"
    else
      n=$(( $(cat "$STATE/failures/$r" 2> /dev/null || echo 0) + 1 ))
      echo "$n" > "$STATE/failures/$r"
      log "$root/$r ($camp, $state): push failed ($n so far); the next round retries"
      failed=$((failed + 1))
    fi
  done
done
[ "$failed" = 0 ] || { log "$failed run(s) not yet on R2"; exit 1; }
