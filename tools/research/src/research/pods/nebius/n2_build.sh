#!/usr/bin/env bash
# Verity's vLLM CPU Builds on vy-nebius-2's idle CPUs (the two nodes are one pool), as `project=verity` fill guests
# that node 2's fill runner pins to its Verity CPUs and freezes (SIGSTOP) through PoUW's timed windows. The Commit stays on node 1.
#
#   on node 1:  n2_build.sh submit KEY ITEM.json|-   (ITEM: the config-run item a lane gives dispatch.py, as in its ready files;
#                                                    `-` reads it from stdin)
#   on node 1:  n2_build.sh stage REPO REV [REPO REV...]   (a checkpoint for node 2's Builds and Commits, staged as submit does)
#   on node 1:  n2_build.sh offload [--loop|--list] (every dispatcher Build Kueue has held 2 min for memory or CPU quota)
#   on node 2:  n2_build.sh run KEY                (the fill job's body; submit writes it)
# Installed at /workspace/verity-guest/bin/n2_build.sh on both nodes.
#
# ITEM is {"template": "config-run", "tree": ..., "env": {ROW, ROLE, REPO, REVISION, SWEEP_DIR, CAMPAIGN, ...},
# "resources": {"build": {"memory": GB}, "gpu": {...}} or "class", optional "queue"/"priority"}. The Build gets the template's
# envs with the item's over them, as dispatch.py gives them; the Commit gets the same item.
#
# submit (as research on node 1):
#   1. materializes the job tree (job_tree.sh, as the job user) and stages it, the checkpoint and the bootstrap stamps on node 2 over
#      the vy-cluster key, never while node 2 has a timed window running or waiting;
#   2. writes the fill job, whose item lives in /workspace/verity-guest/items/KEY.json on node 2.
# run (as research on node 2, inside the fill job):
#   1. the Build, exactly as config-run.yaml's build task runs it: same paths, so every path it records holds on node 1. Node 2's
#      HF_HOME is /workspace/jobs/hf, which on node 1 is a symlink to /workspace/hf; node 2's own /workspace/hf (PoUW's) is untouched;
#   2. sends the row and the run dir back to node 1, publishes the Attempt into node 1's store, and submits the Commit there with
#      node 1's dispatcher (`dispatch.py submit config-run n2-build/KEY --task 1`), which chains the GPU task and supplies R2 custody.
# A rerun after the Build passed (the .n2-build marker) only redoes step 2.
set -euo pipefail
N1=${VY_N1:-vy-n1} N2=${VY_N2:-vy-n2}
GUEST=/workspace/verity-guest
# the fill prio of node 2's vLLM guests: above every other guest's (proofs' are 5 and 10), since circuits' vLLM work goes first
# on both nodes
PRIO=${VY_N2_PRIO:-20}
DISPATCH=${VY_DISPATCH:-/workspace/jobs/dispatch/infra/nebius/dispatch.py}

n2_quiet() { ssh -o ConnectTimeout=10 "$N2" head -1 /workspace/pouw/fill/status.txt | grep -q "timed False, window waiting False"; }

# rsync to node 2, only outside a window; a window that starts stops the transfer, which resumes after it (--partial)
push() {  # push SRC DST [RSYNC-ARG...]
  local src=$1 dst=$2 pid
  shift 2
  while :; do
    until n2_quiet; do sleep 30; done
    nice -n 19 ionice -c3 rsync -a --partial --mkpath --bwlimit=300m "$@" "$src" "$N2:$dst" & pid=$!
    while kill -0 "$pid" 2> /dev/null; do n2_quiet || kill "$pid"; sleep 10; done
    wait "$pid" && return 0
    n2_quiet || continue
    return 1
  done
}

item_get() {  # item_get ITEM.json FIELD...: the item's fields, one per line ("" when absent)
  python3 - "$@" << 'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
e = d.get("env") or {}
for f in sys.argv[2:]:
    v = {"tree": d.get("tree"), "mem": ((d.get("resources") or {}).get("build") or {}).get("memory"),
         "max_min": int(e["BUILD_TIMEOUT"]) // 60 + 10 if e.get("BUILD_TIMEOUT") else None}.get(f, e.get(f))
    print("" if v is None else v)
EOF
}

# the pinned snapshot's files (-L: node 1's snapshots link into its shared blobs/, which stay behind), then a check that node 2 has
# every file as a plain file: the CPU bootstrap's checkpoint check reads the weights
stage_model() {
  local repo=$1 rev=$2 dir
  dir=hub/models--${repo//\//--}/snapshots/$rev
  [ -d "/workspace/hf/$dir" ] || { echo "no checkpoint $repo@$rev on node 1" >&2; return 2; }
  push "/workspace/hf/$dir/" "/workspace/jobs/hf/$dir/" -L
  ssh -n "$N2" "[ -f /workspace/jobs/hf/$dir/config.json ] && [ -z \"\$(find /workspace/jobs/hf/$dir -type l -print -quit)\" ]" \
    || { echo "checkpoint $repo@$rev isn't staged on node 2" >&2; return 3; }
  echo "staged $repo@${rev:0:10} ($(du -sL "/workspace/hf/$dir" | awk '{printf "%.1f GB", $1 / 1e6}'))"
}

submit() {
  local key=$1 item=$2
  [[ $key =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || { echo "KEY: lowercase letters, digits and dashes" >&2; exit 2; }
  if [ "$item" = - ]; then
    mkdir -p "$GUEST/submitted" && item=$GUEST/submitted/$key.json && cat > "$item"
  fi
  local tree row repo rev mem max_min
  { read -r tree; read -r row; read -r repo; read -r rev; read -r mem; read -r max_min; } < <(item_get "$item" tree ROW REPO REVISION mem max_min)
  [ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("template"))' "$item")" = config-run ] \
    || { echo "ITEM: a config-run item" >&2; exit 2; }
  # node 2 caps a guest with MemoryMax at the item's request, in 64-256: node 1's Builds peaked at 4-29 GB against 47-108 GB requests
  # (container_memory_working_set_bytes over 6 h, 2026-09-30 21:15Z)
  # A node-1 request is no limit there, so an MoE item's 66 GB request OOM-killed Qwen3-30B-A3B's Build on node 2 eight times
  # (node2-ops 1 Oct 03:40Z): MoE checkpoints take the cap, the class table's moe row
  [[ $repo =~ -A[0-9]+B|OLMoE|[Mm]o[Ee] ]] && mem=256
  mem=${VY_N2_MEM_GB:-$(( ${mem:-96} < 64 ? 64 : ${mem:-96} > 256 ? 256 : ${mem:-96} ))} max_min=${VY_MAX_MIN:-${max_min:-130}}
  # the queue's placement on node 2, with the item's research question (n2_commit.sh's `place`): a refusal keeps the Build on
  # node 1, and the placement is recorded in node 1's placed.jsonl once the fill job is written
  local question placed cpus=16
  question=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("question") or "")' "$item")
  placed=$(bash "$(dirname "$0")/n2_commit.sh" place build "$key" 0 "$max_min" \
    "${question:-Does $row's Build pass on vy-nebius-2's CPUs, so that node 1 can run its Commit?}" "$cpus" "$mem" kueue-fold) \
    || { echo "not queued" >&2; exit 3; }
  local src
  src=$(sudo -n -u '#1000' bash "$tree/tools/research/src/research/pods/nebius/sky/job_tree.sh" "$tree")
  [ -d "/workspace/hf/hub/models--${repo//\//--}/snapshots/$rev" ] || { echo "no checkpoint $repo@$rev on node 1" >&2; exit 2; }
  push "$src/" "$src/"
  stage_model "$repo" "$rev" || { echo "not queued" >&2; exit 3; }
  push /workspace/jobs/bootstrap/stamps/ /workspace/jobs/bootstrap/stamps/
  push /workspace/jobs/cache/unit-rules/ /workspace/jobs/cache/unit-rules/
  push "$0" "$GUEST/bin/n2_build.sh"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["n2"]={"key":sys.argv[2],"src":sys.argv[3],"mem_gb":int(sys.argv[4]),"cpus":int(sys.argv[5]),"commit_key":sys.argv[6]}; print(json.dumps(d))' \
    "$item" "$key" "$src" "$mem" "$cpus" "${VY_COMMIT_KEY:-n2-build/$key}" | ssh "$N2" "mkdir -p $GUEST/items && rm -f $GUEST/items/$key.done && cat > $GUEST/items/$key.json"
  ssh "$N2" "f=/workspace/pouw/fill/queue/verity-build-$key.sh; cat > \$f.tmp && chmod +x \$f.tmp && mv \$f.tmp \$f" <<EOF
#!/usr/bin/env bash
# fill: owner=kueue-fold gpus=0 max_min=$max_min cpus=$cpus project=verity mem_gb=$mem prio=$PRIO
# Verity vLLM Build $row (node 1 key $key): n2_build.sh on infra/nebius; its Commit runs on node 1.
exec bash $GUEST/bin/n2_build.sh run $key
EOF
  [ -z "$placed" ] || echo "$placed" >> "$GUEST/placed.jsonl"
  echo "queued verity-build-$key on node 2 ($row, ${mem} GB)"
}

# custody: push the Attempt to R2 from a one-off pod on node 1 with the Secret research-r2, so the credentials never leave Kubernetes
push_attempt() {
  local run_id=$1 name
  name=n2-push-$(echo "$run_id" | tr -cd 'a-z0-9-' | tail -c 40)
  ssh "$N1" "KUBECONFIG=\$HOME/.kube/config kubectl delete job $name --ignore-not-found > /dev/null && KUBECONFIG=\$HOME/.kube/config kubectl create -f - > /dev/null \
    && KUBECONFIG=\$HOME/.kube/config kubectl wait --for=condition=complete --timeout=900s job/$name > /dev/null" << EOF
{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "$name", "labels": {"verity.dev/lane": "kueue-fold", "verity.dev/kind": "custody"}},
 "spec": {"backoffLimit": 2, "ttlSecondsAfterFinished": 3600, "template": {"metadata": {"labels": {"verity.dev/lane": "kueue-fold"}}, "spec": {
  "restartPolicy": "Never", "nodeSelector": {"verity.dev/server": "vy-nebius-1"},
  "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
  "securityContext": {"runAsUser": 1000, "runAsGroup": 1000},
  "containers": [{"name": "main", "image": "${VY_DISPATCH_IMAGE:-us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216}",
   "imagePullPolicy": "IfNotPresent",
   "command": ["taskset", "-c", "96-127", "bash", "-c", "cd /workspace/jobs && HOME=/tmp UV_NO_CONFIG=1 PYTHONPATH=$SRC/tools/research/src RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=/workspace/jobs/runs UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data push $run_id"],
   "envFrom": [{"secretRef": {"name": "research-r2"}}],
   "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
   "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}]}],
  "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}}]}}}}
EOF
  echo "$run_id: pushed to R2 ($name)"
}

run() {
  # the vLLM manifests pin every checkpoint at /workspace/hf/hub/..., node 1's cache; node 2's /workspace/hf is PoUW's, so the Build
  # sees /workspace/jobs/hf there in a private mount namespace: nothing is written to PoUW's cache, and the job keeps its scope
  if [ -z "${VY_N2_NS:-}" ]; then
    exec sudo -n unshare --mount --propagation private env VY_N2_NS=1 HOME="$HOME" PATH="$PATH" USER="$USER" \
      bash -c 'mount --bind /workspace/jobs/hf /workspace/hf && exec setpriv --reuid="$SUDO_UID" --regid="$SUDO_GID" --init-groups bash "$0" run "$1"' \
      "$(readlink -f "$0")" "$1"
  fi
  local key=$1 item=$GUEST/items/$1.json
  # a finished job's rerun (the fill runner requeues a job it adopted, whose exit it couldn't read) is a no-op
  if [ ! -e "$item" ] && [ -s "$GUEST/items/$key.done" ]; then
    echo "n2-build $key: already done ($(cat "$GUEST/items/$key.done")); nothing to rerun"
    exit 0
  fi
  [ -e "$item" ] || { echo "n2-build $key: no item $item and no record that it finished" >&2; exit 2; }
  # config-run.yaml's build envs, then the item's over them (dispatch.py's order); HF_HOME is node 2's own (see above)
  export HF_HOME=/workspace/jobs/hf HF_HUB_OFFLINE=1 SWEEP_DIR=/workspace/cp/sweep CUDA_HOME=/workspace/jobs/cuda-12.9 \
    TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions PY=/workspace/jobs/venv312/bin/python PY312=/workspace/jobs/venv312/bin/python \
    RESEARCH_PY=/workspace/jobs/bin/py312 RESEARCH_RUNS=/workspace/jobs/runs RESEARCH_STORE=/workspace/jobs/store \
    HOT_ROOT=/workspace/jobs/hot HOT_RELEASING_MARKER=/workspace/jobs/hot/releasing HOT_RELEASE_JSON=/workspace/jobs/hot/release.json \
    GUARD_PRESERVE_FILE=/workspace/jobs/dm/guard.preserve BUILD_JOBS=auto \
    VERITY_UNIT_RULE_CACHE=/workspace/jobs/cache/unit-rules UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python \
    CAMPAIGN=nebius-sweep
  # node 2's store has no remote, so `research run` refuses without this (n2_commit.sh's load_item says why it is a variable)
  export RESEARCH_LOCAL_ONLY="node 2's n2-custody timer (n2_custody.sh) takes every ended run under /workspace/jobs/runs to node 1's store and R2"
  eval "$(python3 - "$item" << 'EOF'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
env = {k: str(v) for k, v in (d.get("env") or {}).items() if k != "HF_HOME"}
n2 = d["n2"]
env.update(SRC=n2["src"], TREE=d["tree"], KEY=n2["key"], COMMIT_KEY=n2.get("commit_key") or f"n2-build/{n2['key']}",
           BUILD_RAM_BUDGET_GB=str(n2["mem_gb"]), OMP_NUM_THREADS=str(n2["cpus"]))
print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()))
print("COMMIT_ARGS=(" + " ".join(shlex.quote(a) for a in
      [x for k, v in (d.get("env") or {}).items() for x in ("--env", f"{k}={v}")]
      + ([ "--class", d["class"]] if d.get("class") else [])
      + (["--resources", json.dumps(d["resources"])] if d.get("resources") else [])
      + (["--queue", d["queue"]] if d.get("queue") else []) + (["--priority", d["priority"]] if d.get("priority") else [])
      + (["--lease", d["lease"]] if d.get("lease") else [])
      + (["--template-sha", d["template_sha"]] if d.get("template_sha") else [])) + ")")
EOF
)"
  local row_dir=$SWEEP_DIR/$ROW marker=$SWEEP_DIR/$ROW/.n2-build run_id
  mkdir -p /workspace/jobs/hot /workspace/jobs/dm /workspace/jobs/cache/unit-rules "$row_dir"
  if [ -s "$marker" ]; then
    run_id=$(cat "$marker")
  else
    [ -x /workspace/jobs/bin/py312 ] || printf '#!/bin/sh\nexec /workspace/jobs/bin/uv run --no-project --python 3.12 python "$@"\n' > /workspace/jobs/bin/py312
    (cd "$SRC/integrations/vllm" && bash "$SRC/tools/research/src/research/pods/nebius/sky/vllm_bootstrap.sh" --cases "$ROLE" --cpu \
      --out "/workspace/jobs/bootstrap/$ROLE-cpu")
    export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH PYTHONPATH=$SRC/integrations/vllm:$SRC/tools/research/src CUDA_VISIBLE_DEVICES=
    export LD_LIBRARY_PATH=/workspace/jobs/cuda-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    run_id=r$(date -u +%Y%m%d-%H%M%S)-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')
    echo "$run_id" > "$row_dir/two-task-build-run"
    /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research run --id "$run_id" --tool vllm.build --project verity \
      --cwd "$SRC" --campaign "$CAMPAIGN" -- "$PY" -m verity_vllm.pipeline.cli row stage build "$ROW" "$ROLE" "$REPO" "$REVISION" \
      --config-run 1 --replay-k 460
    echo "$run_id" > "$marker"
  fi
  # back to node 1: the row, the run dir, the Attempt in node 1's store, then the Commit through node 1's dispatcher
  local sudo_rsync=(--rsync-path="sudo -n rsync" --chown=1000:1000 --mkpath)
  rsync -a "${sudo_rsync[@]}" "$row_dir/" "$N1:$row_dir/"
  rsync -a "${sudo_rsync[@]}" "/workspace/jobs/runs/$run_id/" "$N1:/workspace/jobs/runs/$run_id/"
  ssh "$N1" "cd /workspace/jobs && sudo -n -u '#1000' env UV_NO_CONFIG=1 RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=/workspace/jobs/runs PYTHONPATH=$SRC/tools/research/src \
    UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python \
    /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data attempt publish /workspace/jobs/runs/$run_id"
  push_attempt "$run_id"
  ssh "$N1" "KUBECONFIG=\$HOME/.kube/config /workspace/jobs/venv312/bin/python $DISPATCH submit config-run $COMMIT_KEY --tree $TREE --task 1 \
    $(printf '%q ' "${COMMIT_ARGS[@]}")"
  # node 1 holds the row, the run and the Attempt now (node2-ops: delete a Build's outputs once they're back); the tree stays for reuse
  echo "$run_id $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$GUEST/items/$KEY.done"
  rm -rf "$row_dir" "/workspace/jobs/runs/$run_id" "$item"
  echo "n2-build $KEY: Build $run_id on node 2, Commit submitted on node 1"
}

# the dispatcher's Builds that Kueue holds for memory or CPU quota, one per line: JOB KEY N2KEY, their items in $1/N2KEY.json
held_builds() {
  KUBECONFIG=$HOME/.kube/config kubectl get jobs -l verity.dev/dispatch=node1 -o json > "$1/k8s.json"
  KUBECONFIG=$HOME/.kube/config kubectl get workloads -o json > "$1/wl.json"
  python3 - "$1" "${VY_OFFLOAD_WAIT_S:-120}" << 'EOF'
import calendar, json, re, sys, time
d, wait = sys.argv[1], int(sys.argv[2])
jobs = {j["metadata"]["name"]: j for j in json.load(open(f"{d}/k8s.json"))["items"]}
for w in json.load(open(f"{d}/wl.json"))["items"]:
    st = {c["type"]: c for c in (w.get("status") or {}).get("conditions") or []}
    held = st.get("QuotaReserved", {})
    if st.get("Finished", {}).get("status") == "True" or held.get("status") != "False":
        continue
    if not re.search(r"insufficient unused quota for (memory|cpu)\b", held.get("message", "")):
        continue
    since = calendar.timegm(time.strptime(w["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"))
    job = jobs.get(next((o["name"] for o in w["metadata"].get("ownerReferences") or [] if o["kind"] == "Job"), ""))
    if job is None or time.time() - since < wait or not job["spec"].get("suspend"):
        continue
    a, lab = job["metadata"]["annotations"], job["metadata"]["labels"]
    item = json.loads(a["verity.dev/item-json"])
    if item.get("template") != "config-run" or lab.get("verity.dev/task") != "0":
        continue
    n2key = re.sub(r"[^a-z0-9-]+", "-", a["verity.dev/item"].lower()).strip("-")[:63]
    json.dump(item, open(f"{d}/{n2key}.json", "w"))
    print(job["metadata"]["name"], a["verity.dev/item"], n2key)
EOF
}

# move each held Build to node 2 (the Commit comes back under the dispatcher's own key), while node 2 has fewer than
# VY_N2_MAX_QUEUED Verity Builds waiting; `offload --loop` repeats every minute (unit vy-n2-offload on node 1, as research)
offload() {
  case "${1:-}" in
    --loop) while :; do offload || true; sleep "${VY_OFFLOAD_EVERY_S:-60}"; done ;;
    --list) (tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; held_builds "$tmp"); return ;;
  esac
  (
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    n2_quiet || exit 0
    while read -r job key n2key; do
      queued=$(ssh -n "$N2" 'ls /workspace/pouw/fill/queue/ | grep -c "^verity-build-"' || true)
      [ "${queued:-0}" -lt "${VY_N2_MAX_QUEUED:-6}" ] || { echo "node 2 has $queued Verity Builds waiting; $job stays"; break; }
      VY_COMMIT_KEY=$key bash "$0" submit "$n2key" "$tmp/$n2key.json" < /dev/null \
        || { echo "$job: submit failed, left on node 1" >&2; continue; }
      if [ "$(KUBECONFIG=$HOME/.kube/config kubectl get job "$job" -o jsonpath='{.spec.suspend}')" = true ] \
        && KUBECONFIG=$HOME/.kube/config kubectl delete job "$job" > /dev/null; then
        printf '{"ev": "moved", "job": "%s", "key": "%s", "to": "vy-nebius-2", "n2_key": "%s", "t": "%s"}\n' \
          "$job" "$key" "$n2key" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /workspace/jobs/dispatch/log.jsonl
        echo "moved $key ($job) to node 2"
      else
        ssh -n "$N2" "rm -f /workspace/pouw/fill/queue/verity-build-$n2key.sh $GUEST/items/$n2key.json"
        echo "$job was admitted meanwhile; it stays on node 1"
      fi
    done < <(held_builds "$tmp")
  )
}

cmd=${1:-}; shift || true
case "$cmd" in
  submit) submit "$@" ;;
  stage) while [ $# -ge 2 ]; do stage_model "$1" "$2" || true; shift 2; done ;;
  run) run "$@" ;;
  offload) offload "$@" ;;
  *) sed -n '2,6p' "$0" >&2; exit 2 ;;
esac
