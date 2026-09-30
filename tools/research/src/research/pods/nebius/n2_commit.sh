#!/usr/bin/env bash
# Verity's vLLM Commits (config-run task 1) on vy-nebius-2's idle GPUs, as `gpus=1 project=verity` fill guests: node 2's fill runner
# starts one only when no PoUW GPU job is ready and at least 2 GPUs are free, preempts it first, and stops it in every timed window
# (it is requeued and reruns from the top). The Build stays where it ran; the CPU replay goes back to node 1's dispatcher.
#
#   on node 1:  n2_commit.sh submit KEY ITEM.json|- [NODE1_JOB]   (ITEM: the config-run item, as in a Job's verity.dev/item-json;
#                                                                 NODE1_JOB: its queued Commit Job, deleted once node 2 has it)
#   on node 1:  n2_commit.sh item NODE1_JOB                       (prints that Job's item)
#   on node 2:  n2_commit.sh run KEY                              (the fill job's body; submit writes it)
# Installed at /workspace/verity-guest/bin/n2_commit.sh on both nodes; n2_build.sh's sibling.
#
# submit (as research on node 1) refuses a row that isn't TP1, whose Build hasn't passed on node 1 (no manifest.json or Build run),
# or whose checkpoint REPO@REVISION isn't staged on node 2 as plain files (/workspace/jobs/hf, or PoUW's /workspace/hf, which is only
# read: its snapshot is copied into /workspace/jobs/hf). It stages the job tree, the Build row (Programs, manifest), the bootstrap
# stamps and the tree's Triton/vLLM caches on node 2 at node 1's paths, never while node 2 has a timed window running or waiting,
# resolves the Build's artifact in node 1's store, and writes the fill job.
# run (as research on node 2, inside the fill job) runs config-run.yaml's gpu task with REPLAY_DEFERRED=1, then sends the row and
# the run dir back to node 1, publishes the Attempt into node 1's store, preserves it in R2 (from node 2 when node 2 has the R2
# environment and the run dir is under 64 MiB, else from a pod on node 1 with the Secret research-r2), and submits the replay with
# node 1's dispatcher (`dispatch.py submit config-run COMMIT_KEY --task 2`). A Commit that fails on node 2 goes back to node 1's
# queue (`--task 1`) after its Attempt is custodied. A rerun after the Commit passed (the .n2-commit marker) only redoes the return.
set -euo pipefail
N1=${VY_N1:-vy-n1} N2=${VY_N2:-vy-n2}
GUEST=/workspace/verity-guest
DISPATCH=${VY_DISPATCH:-/workspace/jobs/dispatch/infra/nebius/dispatch.py}
OWNER=${VY_N2_COMMIT_OWNER:-bc-698052e1-8e2b-5736-997e-8fa229b85b3c}
KUBE_DISPATCH="KUBECONFIG=\$HOME/.kube/config"
KUBE="$KUBE_DISPATCH kubectl"
RESEARCH_N1="cd /workspace/jobs && sudo -n -u '#1000' env UV_NO_CONFIG=1 RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=/workspace/jobs/runs \
  UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python"

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
    v = {"tree": d.get("tree"), "template": d.get("template"), "SWEEP_DIR": e.get("SWEEP_DIR") or "/workspace/cp/sweep"}.get(f, e.get(f))
    print("" if v is None else v)
EOF
}

# the checks submit makes before it touches either node
check_item() {  # check_item KEY ITEM.json: exit 2 unless KEY is a key and ITEM a TP1 config-run item with ROW, ROLE, REPO, REVISION
  local key=$1 item=$2 template row role repo rev
  [[ $key =~ ^[a-z0-9][a-z0-9-]{0,55}$ ]] || { echo "KEY: lowercase letters, digits and dashes" >&2; exit 2; }
  { read -r template; read -r row; read -r role; read -r repo; read -r rev; } < <(item_get "$item" template ROW ROLE REPO REVISION)
  [ "$template" = config-run ] || { echo "ITEM: a config-run item" >&2; exit 2; }
  [ -n "$row" ] && [ -n "$role" ] && [ -n "$repo" ] && [ -n "$rev" ] || { echo "ITEM: needs ROW, ROLE, REPO and REVISION" >&2; exit 2; }
  [[ $row == *__tp1__* ]] || { echo "ITEM: $row isn't TP1; a node-2 guest has one GPU" >&2; exit 2; }
}

item() { ssh "$N1" "$KUBE get job $1 -o jsonpath='{.metadata.annotations.verity\.dev/item-json}'"; echo; }

submit() {
  local key=$1 item=$2 job=${3:-}
  if [ "$item" = - ]; then
    mkdir -p "$GUEST/submitted" && item=$GUEST/submitted/commit-$key.json && cat > "$item"
  fi
  check_item "$key" "$item"
  local tree row role repo rev sweep
  { read -r tree; read -r row; read -r role; read -r repo; read -r rev; read -r sweep; } \
    < <(item_get "$item" tree ROW ROLE REPO REVISION SWEEP_DIR)
  local row_dir=$sweep/$row model=models--${repo//\//--}
  [ -s "$row_dir/manifest.json" ] && [ -s "$row_dir/two-task-build-run" ] \
    || { echo "no passed Build in $row_dir on node 1 (manifest.json, two-task-build-run); not queued" >&2; exit 3; }
  # staged means plain files on node 2; PoUW's /workspace/hf is read, never written
  ssh -n "$N2" "d=/workspace/jobs/hf/hub/$model/snapshots/$rev; p=/workspace/hf/hub/$model/snapshots/$rev
    if ! { [ -f \$d/config.json ] && [ -z \"\$(find \$d -type l -print -quit)\" ]; } && [ -f \$p/config.json ]; then
      mkdir -p \$d.tmp && cp -rL \$p/. \$d.tmp/ && rm -rf \$d && mv \$d.tmp \$d
    fi
    [ -f \$d/config.json ] && [ -z \"\$(find \$d -type l -print -quit)\" ]" \
    || { echo "checkpoint $repo@$rev isn't staged on node 2 (n2_build.sh stage REPO REV); not queued" >&2; exit 3; }
  local src build_run build_art
  # the Commit keeps node 1's dispatcher key, so its replay and records chain on under it
  [ -z "$job" ] || [ -n "${VY_COMMIT_KEY:-}" ] || VY_COMMIT_KEY=$(ssh -n "$N1" "$KUBE get job $job -o jsonpath='{.metadata.annotations.verity\\.dev/item}'")
  src=$(sudo -n -u '#1000' bash "$tree/tools/research/src/research/pods/nebius/sky/job_tree.sh" "$tree")
  build_run=$(cat "$row_dir/two-task-build-run")
  build_art=$(cd /workspace/jobs && sudo -n -u '#1000' env UV_NO_CONFIG=1 RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=/workspace/jobs/runs \
    PYTHONPATH="$src/tools/research/src" UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python \
    /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data show "$build_run" --json 2> /dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d, list) else d; print((d.get("outputs") or {}).get("build", ""))' \
    2> /dev/null || true)
  push "$src/" "$src/"
  push "$row_dir/" "$row_dir/" --exclude=/commit/ --exclude=/pipeline-commit-run --exclude=/.n2-commit
  push /workspace/jobs/bootstrap/stamps/ /workspace/jobs/bootstrap/stamps/
  local c
  for c in triton vllm; do
    # a warm start only: some cache files are uid 1000's alone, and a Commit without them compiles its own (~5 min)
    [ ! -d "/workspace/jobs/cache/$c/$(basename "$src")" ] || push "/workspace/jobs/cache/$c/$(basename "$src")/" "/workspace/jobs/cache/$c/$(basename "$src")/" \
      || echo "the $c cache of $(basename "$src") went over in part" >&2
  done
  push "$0" "$GUEST/bin/n2_commit.sh"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d.setdefault("env", {})["REPLAY_DEFERRED"]="1"
d["n2"]={"key":sys.argv[2],"src":sys.argv[3],"build_run":sys.argv[4],"build_art":sys.argv[5],"commit_key":sys.argv[6]}; print(json.dumps(d))' \
    "$item" "$key" "$src" "$build_run" "$build_art" "${VY_COMMIT_KEY:-n2-commit/$key}" \
    | ssh "$N2" "mkdir -p $GUEST/items && cat > $GUEST/items/commit-$key.json"
  ssh "$N2" "f=/workspace/pouw/fill/queue/verity-commit-$key.sh; cat > \$f.tmp && chmod +x \$f.tmp && mv \$f.tmp \$f" <<EOF
#!/usr/bin/env bash
# fill: owner=$OWNER gpus=1 max_min=30 cpus=8 project=verity
# Verity vLLM Commit $row (node 1 key $key): n2_commit.sh on infra/nebius; its replay runs on node 1.
exec bash $GUEST/bin/n2_commit.sh run $key
EOF
  if [ -n "$job" ]; then
    # node 1's queued Commit goes only while Kueue still holds it; if it was admitted meanwhile, node 2's copy goes instead
    if [ "$(ssh -n "$N1" "$KUBE get job $job -o jsonpath='{.spec.suspend}'")" = true ] && ssh -n "$N1" "$KUBE delete job $job" > /dev/null; then
      printf '{"ev": "moved", "job": "%s", "key": "%s", "to": "vy-nebius-2", "n2_key": "commit-%s", "t": "%s"}\n' \
        "$job" "${VY_COMMIT_KEY:-n2-commit/$key}" "$key" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | ssh "$N1" "cat >> /workspace/jobs/dispatch/log.jsonl"
    else
      ssh -n "$N2" "rm -f /workspace/pouw/fill/queue/verity-commit-$key.sh $GUEST/items/commit-$key.json"
      echo "$job was admitted on node 1 meanwhile; not queued on node 2" >&2
      exit 4
    fi
  fi
  echo "queued verity-commit-$key on node 2 ($row, Build $build_run ${build_art:-without an artifact})"
}

# custody: push the Attempt to R2 from a one-off pod on node 1 with the Secret research-r2, so the credentials never leave Kubernetes
push_attempt() {
  local run_id=$1 src=$2 name
  name=n2-push-$(echo "$run_id" | tr -cd 'a-z0-9-' | tail -c 40)
  ssh "$N1" "$KUBE delete job $name --ignore-not-found > /dev/null && $KUBE create -f - > /dev/null \
    && $KUBE wait --for=condition=complete --timeout=900s job/$name > /dev/null" << EOF
{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "$name", "labels": {"verity.dev/lane": "n2-commits", "verity.dev/kind": "custody"}},
 "spec": {"backoffLimit": 2, "ttlSecondsAfterFinished": 3600, "template": {"metadata": {"labels": {"verity.dev/lane": "n2-commits"}}, "spec": {
  "restartPolicy": "Never", "nodeSelector": {"verity.dev/server": "vy-nebius-1"},
  "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
  "securityContext": {"runAsUser": 1000, "runAsGroup": 1000},
  "containers": [{"name": "main", "image": "${VY_DISPATCH_IMAGE:-us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216}",
   "imagePullPolicy": "IfNotPresent",
   "command": ["taskset", "-c", "96-127", "bash", "-c", "cd /workspace/jobs && HOME=/tmp UV_NO_CONFIG=1 PYTHONPATH=$src/tools/research/src RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=/workspace/jobs/runs UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data push $run_id"],
   "envFrom": [{"secretRef": {"name": "research-r2"}}],
   "resources": {"requests": {"cpu": "1", "memory": "2Gi"}},
   "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}]}],
  "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}}]}}}}
EOF
  echo "$run_id: pushed to R2 from node 1 ($name)"
}

run() {
  # the manifests pin every checkpoint at /workspace/hf/hub/..., node 1's cache; node 2's /workspace/hf is PoUW's, so the Commit sees
  # /workspace/jobs/hf there in a private mount namespace (as n2_build.sh's Build). sudo drops the environment: the lease's GPU goes along
  if [ -z "${VY_N2_NS:-}" ]; then
    exec sudo -n unshare --mount --propagation private env VY_N2_NS=1 HOME="$HOME" PATH="$PATH" USER="$USER" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
      FILL="${FILL:-}" FILL_JOB="${FILL_JOB:-}" GPU_LEASE_WHO="${GPU_LEASE_WHO:-}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
      bash -c 'mount --bind /workspace/jobs/hf /workspace/hf && exec setpriv --reuid="$SUDO_UID" --regid="$SUDO_GID" --init-groups bash "$0" run "$1"' \
      "$(readlink -f "$0")" "$1"
  fi
  local key=$1 item=$GUEST/items/commit-$1.json
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no GPU leased (CUDA_VISIBLE_DEVICES is empty)" >&2; exit 2; }
  # config-run.yaml's gpu envs, then the item's over them (dispatch.py's order); HF_HOME is node 2's own (see above)
  export HF_HOME=/workspace/jobs/hf HF_HUB_OFFLINE=1 SWEEP_DIR=/workspace/cp/sweep CUDA_HOME=/workspace/jobs/cuda-12.9 \
    TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions PY=/workspace/jobs/venv312/bin/python PY312=/workspace/jobs/venv312/bin/python \
    RESEARCH_PY=/workspace/jobs/bin/py312 RESEARCH_RUNS=/workspace/jobs/runs RESEARCH_STORE=/workspace/jobs/store \
    HOT_ROOT=/workspace/jobs/hot HOT_RELEASING_MARKER=/workspace/jobs/hot/releasing HOT_RELEASE_JSON=/workspace/jobs/hot/release.json \
    GUARD_PRESERVE_FILE=/workspace/jobs/dm/guard.preserve OV_NOISY=true \
    UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python CAMPAIGN=nebius-sweep
  eval "$(python3 - "$item" << 'EOF'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
env = {k: str(v) for k, v in (d.get("env") or {}).items() if k != "HF_HOME"}
n2 = d["n2"]
env.update(SRC=n2["src"], TREE=d["tree"], KEY=n2["key"], COMMIT_KEY=n2["commit_key"], BUILD_ART=n2.get("build_art") or "",
           REPLAY_DEFERRED="1")
print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()))
print("ITEM_ARGS=(" + " ".join(shlex.quote(a) for a in
      [x for k, v in {**(d.get("env") or {}), "REPLAY_DEFERRED": "1"}.items() for x in ("--env", f"{k}={v}")]
      + ([ "--class", d["class"]] if d.get("class") else [])
      + (["--resources", json.dumps(d["resources"])] if d.get("resources") else [])
      + (["--queue", d["queue"]] if d.get("queue") else []) + (["--priority", d["priority"]] if d.get("priority") else [])) + ")")
EOF
)"
  local row_dir=$SWEEP_DIR/$ROW marker=$SWEEP_DIR/$ROW/.n2-commit run_id rc=0 research
  research=(/workspace/jobs/bin/uv run --no-project --python 3.12 python -m research)
  mkdir -p /workspace/jobs/hot /workspace/jobs/dm
  nvidia-smi --query-gpu=index,name,driver_version,clocks.gr,clocks.sm,clocks.mem,clocks.applications.graphics,power.limit \
    --format=csv -i "$CUDA_VISIBLE_DEVICES" || true
  if [ -s "$marker" ]; then
    read -r run_id rc < "$marker"
  else
    [ -x /workspace/jobs/bin/py312 ] || printf '#!/bin/sh\nexec /workspace/jobs/bin/uv run --no-project --python 3.12 python "$@"\n' > /workspace/jobs/bin/py312
    (cd "$SRC/integrations/vllm" && bash "$SRC/tools/research/src/research/pods/nebius/sky/vllm_bootstrap.sh" --cases "$ROLE" \
      --out "/workspace/jobs/bootstrap/$ROLE")
    export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH PYTHONPATH=$SRC/integrations/vllm:$SRC/tools/research/src
    case "$ROW" in *__bi-eager) export TRITON_CACHE_DIR=/workspace/jobs/cache/triton/$(basename "$SRC") VLLM_CACHE_ROOT=/workspace/jobs/cache/vllm/$(basename "$SRC");; esac
    if grep -q staged_sources "$SRC/integrations/vllm/verity_vllm/commit/committer/native_jit.py"; then
      export HIDDEN_GPU_BUILD=/workspace/jobs/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=/workspace/jobs/cache/jit/nc
    fi
    echo "commit cites the Build: artifact ${BUILD_ART:-none}"
    run_id=r$(date -u +%Y%m%d-%H%M%S)-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')
    echo "$run_id" > "$row_dir/pipeline-commit-run"
    "${research[@]}" run --id "$run_id" --tool vllm.commit --project verity --cwd "$SRC" --campaign "$CAMPAIGN" \
      ${BUILD_ART:+--input "build=$BUILD_ART"} -- "$PY" -m verity_vllm.pipeline.cli row stage commit "$ROW" "$ROLE" "$REPO" "$REVISION" \
      --config-run 1 --replay-k 460 --replay-deferred 1 || rc=$?
    # a stop (the lease's max_min, a waiter, a window) is requeued and reruns the Commit from the top
    case "$rc" in 124 | 137 | 143) exit 143 ;; esac
    echo "$run_id $rc" > "$marker"
  fi
  # back to node 1: the row, the run dir, the Attempt in node 1's store and in R2, then the replay (or, on failure, the Commit) there
  local sudo_rsync=(--rsync-path="sudo -n rsync" --chown=1000:1000 --mkpath)
  rsync -a "${sudo_rsync[@]}" "$row_dir/" "$N1:$row_dir/"
  rsync -a "${sudo_rsync[@]}" "/workspace/jobs/runs/$run_id/" "$N1:/workspace/jobs/runs/$run_id/"
  ssh "$N1" "$RESEARCH_N1 PYTHONPATH=$SRC/tools/research/src /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data attempt publish /workspace/jobs/runs/$run_id"
  if [ -n "${R2_ENDPOINT:-}" ] && [ "$(du -sm "/workspace/jobs/runs/$run_id" | cut -f1)" -lt 64 ]; then
    PYTHONPATH=$SRC/tools/research/src "${research[@]}" data push "$run_id" && echo "$run_id: pushed to R2 from node 2"
  else
    push_attempt "$run_id" "$SRC"
  fi
  local task=2
  [ "$rc" = 0 ] || task=1
  ssh "$N1" "$KUBE_DISPATCH /workspace/jobs/venv312/bin/python $DISPATCH submit config-run $COMMIT_KEY --tree $TREE --task $task \
    $(printf '%q ' "${ITEM_ARGS[@]}")"
  # node 1 holds the row, the run and the Attempt now; the tree stays for reuse
  rm -rf "$row_dir" "/workspace/jobs/runs/$run_id" "$item"
  echo "n2-commit $KEY: Commit $run_id on node 2 rc $rc; task $task submitted on node 1"
}

cmd=${1:-}; shift || true
case "$cmd" in
  submit) submit "$@" ;;
  item) item "$@" ;;
  run) run "$@" ;;
  check) check_item "$@" ;;
  *) sed -n '5,9p' "$0" >&2; exit 2 ;;
esac
