#!/usr/bin/env bash
# Verity's vLLM Commits (config-run task 1) on vy-nebius-2's idle GPUs, as `gpus=1` (a TP2 row: `gpus=2`) `project=verity` fill
# guests with max_min VY_N2_COMMIT_MAX_MIN (40: node 2's gaps between booked windows run 5-60 min; commit_max_min gives a long row
# more): node 2's fill runner starts them ahead of its other GPU fill, never into a booked
# window, and stops them for a waiter or a timed window (requeued, rerun from the top). The Build stays where it ran; the CPU
# replay runs on node 2's Verity CPUs (48-95) as a `gpus=0` guest.
#
#   on node 1:  n2_commit.sh submit KEY ITEM.json|- [NODE1_JOB]   (ITEM: the config-run item, as in a Job's verity.dev/item-json;
#                                                                 NODE1_JOB: its queued Commit Job, deleted once node 2 has it)
#   on node 1:  n2_commit.sh item NODE1_JOB                       (prints that Job's item)
#   on node 1:  n2_commit.sh offload [--loop|--list]             (every Commit Kueue has held 2 min whose checkpoint node 2 has;
#                                                                 VY_N2_FIRST=0: only while deployments-gpu has no free GPU)
#   on node 1:  n2_commit.sh n1-free                              (the GPUs deployments-gpu could still admit)
#   on node 1:  n2_commit.sh place KIND KEY GPUS MAX_MIN QUESTION (the queue's placement on node 2, as submit records it; a dry run)
#   on node 1:  n2_commit.sh max-min ROW                          (the max_min submit gives ROW's guest)
#   on node 1:  n2_commit.sh plan-keyed JOB_TREE ROW_DIR          (whether the row's plans are keyed to the code JOB_TREE runs)
#   on node 2:  n2_commit.sh run KEY | replay KEY                 (the fill jobs' bodies; submit and run write them)
#   on node 2:  n2_commit.sh fetch-manifests ART...              (the manifests of ART... that node 2's store lacks, from node 1's)
# Installed at /workspace/verity-guest/bin/n2_commit.sh on both nodes; n2_build.sh's sibling.
#
# submit (as research on node 1) refuses a row that isn't TP1 or TP2, whose Build hasn't passed on node 1 (no manifest.json or Build run),
# whose plans are keyed to other code than the job tree its Commit would run (plan_keyed: that Commit would derive them with node
# 2's GPU held), or whose checkpoint REPO@REVISION isn't staged on node 2 as plain files (/workspace/jobs/hf, or PoUW's
# /workspace/hf, which is only read: its snapshot is copied into /workspace/jobs/hf). It stages the job tree, the Build row
# (Programs, manifest), the bootstrap stamps and the tree's Triton/vLLM caches on node 2 at node 1's paths, never while node 2 has
# a timed window running or waiting, resolves the Build's artifact in node 1's store, copies that artifact's store manifest into
# node 2's store (which has no remote, and the Commit's precheck reads it), and writes the fill job. Before any of that it asks the
# queue (`cluster submit`, from VY_QUEUE_SRC's infra/cluster) to place the Commit on node 2 with its research question: a refusal
# keeps the row on node 1, and once the fill job is written the placement is appended to node 1's $GUEST/placed.jsonl. Node 2's
# fill queue stays the executor, so these count apart from `research run --queue` (top-level 5:14 AM PDT). A queue that gives no
# answer doesn't hold the row back.
# run (as research on node 2, inside the fill job) runs config-run.yaml's gpu task with REPLAY_DEFERRED=1, records the GPU's
# driver, clocks and vLLM version in the row (n2_gpu_env.json), then sends the row (without the replay bundle) and the run dir back
# to node 1, publishes the Attempt into node 1's store, preserves it in R2 (from node 2 when node 2 has the R2 environment and the
# run dir is under 64 MiB, else from a pod on node 1 with the Secret research-r2), and queues the replay on node 2. replay runs
# config-run.yaml's replay task there and custodies it the same way. A Commit or replay that fails on node 2 goes back to node 1's
# dispatcher (`dispatch.py submit config-run COMMIT_KEY --task 1|2`) with the whole row, and offload never moves a Commit that
# went back to node 2 again; VY_N2_REPLAY=n1 sends every replay there.
# A rerun after a stage finished (the .n2-commit / .n2-replay marker) only redoes the return.
set -euo pipefail
N1=${VY_N1:-vy-n1} N2=${VY_N2:-vy-n2}
GUEST=/workspace/verity-guest
# the fill prio of node 2's vLLM guests: above every other guest's (proofs' are 5 and 10), since circuits' vLLM work goes first
# on both nodes (Daniel 1 Oct 01:35Z)
PRIO=${VY_N2_PRIO:-20}
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
  [[ $row =~ __tp[12]__ ]] || { echo "ITEM: $row isn't TP1 or TP2; a node-2 guest has one or two GPUs" >&2; exit 2; }
}

tp_of() { [[ $1 =~ __tp([12])__ ]] && echo "${BASH_REMATCH[1]}"; }

# A row with 1024 input tokens or batch 32+ ran 20-120 min on node 1 by 3 Oct (gemma2-9b B8 67-76, qwen3-30b-a3b 44-54, qwen3-14b
# TP2 B8 ~45), and a stop at max_min reruns it from the top, so VY_N2_COMMIT_MAX_MIN (25 for node 2's 30-min gaps on 1 Oct) killed
# cov-gm359 twice. Such a row gets VY_N2_COMMIT_LONG_MAX_MIN (the fill runner's cap for a Commit guest, FILL_COMMIT_MAX_MIN), which
# it starts only where no booked window is that close.
commit_max_min() {  # commit_max_min ROW
  local b=0
  [[ $1 =~ __b([0-9]+)__ ]] && b=${BASH_REMATCH[1]}
  if [[ $1 == *__i1024__* ]] || [ "$b" -ge 32 ]; then echo "${VY_N2_COMMIT_LONG_MAX_MIN:-120}"; else echo "${VY_N2_COMMIT_MAX_MIN:-40}"; fi
}

item() { ssh "$N1" "$KUBE get job $1 -o jsonpath='{.metadata.annotations.verity\.dev/item-json}'"; echo; }

# node 2's store has no remote, so a stage's precheck reads the manifests it cites (the Build's; the replay's Commit) from node 2's
# own store: submit copies them there, and run and replay fetch any still missing from node 1's. A manifest's sha256 is its id
STORE=${VY_STORE:-/workspace/jobs/store}
art_hex() {  # art_hex ART: its 64 hex digits, or exit 2
  [[ $1 =~ ^art:([0-9a-f]{64})$ ]] || { echo "not an art: id: $1" >&2; exit 2; }
  echo "${BASH_REMATCH[1]}"
}

fetch_manifests() {  # fetch_manifests ART...: (on node 2) every manifest node 2's store lacks, from node 1's, checked against its id
  local art hex f tmp
  for art in "$@"; do
    [ -n "$art" ] || continue
    hex=$(art_hex "$art") f=$STORE/manifests/$hex.json
    [ -s "$f" ] && continue
    tmp=$STORE/manifests/.n2-$hex-$$
    if rsync -a "$N1:$STORE/manifests/$hex.json" "$tmp" && [ "$(sha256sum < "$tmp" | cut -d' ' -f1)" = "$hex" ]; then
      mv "$tmp" "$f" && echo "fetched the manifest of $art from node 1's store"
    else
      rm -f "$tmp"
      echo "the manifest of $art isn't in node 1's store (or doesn't hash to its id); the precheck will refuse it" >&2
    fi
  done
}

n2_question() {  # n2_question ROW: node 2's question for a Commit of ROW (the Attempt's second `question` label, load_item's)
  printf "Does %s's Commit validate on vy-nebius-2's RTX PRO 6000, its replay equal on every sampled unit, as node 1's Commits of its Build do?" "$1"
}

place() {  # place KIND KEY GPUS MAX_MIN QUESTION [CPUS [MEM_GB [SUBMITTER]]]: the queue's placement on node 2 as one placed.jsonl
  # line (none when it gives no answer); exit 3 when it refuses the job. n2_build.sh's submit calls it too.
  local src=${VY_QUEUE_SRC:-$GUEST/cluster-src} out mem=()
  [ -z "${7:-}" ] || mem=(--mem-gb "$7")
  out=$(PYTHONPATH=$src/src python3 -m cluster submit "$src/descriptions/nebius.toml" --run-id "n2-$1-$2" --now "$(date +%s)" \
    --project verity --submitter "${8:-$OWNER}" --question "$5" --gpus "$3" --cpus "${6:-8}" "${mem[@]}" --max-s $(($4 * 60)) \
    --preempt requeue --node vy-nebius-2 2> /dev/null | tail -1) || true
  python3 - "$1" "$2" "$5" "$out" << 'EOF'
import json, sys, time
kind, key, question, out = sys.argv[1:]
try:
    placed = json.loads(out)
except ValueError:
    print(f"the queue gave no placement for {kind} {key}", file=sys.stderr)
    sys.exit(0)
if "refused" in placed:
    print(f"the queue refuses {kind} {key}: {placed['refused']}", file=sys.stderr)
    sys.exit(3)
print(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, "key": key, "node": placed.get("node"),
                  "why": placed.get("why"), "question": placed.get("question") or question, "executor": "fill-queue"}))
EOF
}

# A row's plans (plan/commit_plan.json, call_boundary_plan.json) carry the code identity of the job tree its Build ran
# (commit_plan.key). A Commit from a job tree of other code derives both again, which node 1's leased Commit does before it takes a
# GPU and a node 2 guest does with its GPU held: 5.0 of node 2's 9.2 idle Commit GPU-h in the 24 h to 2 Oct 16:30Z, every time
# for that reason alone (cov-gm176 held GPU 6 for two 25-min leases at 0%, built before the 14:17Z redeploy). The key's other
# fields are the row's own files, which go to node 2 unchanged; the code identity covers the workloads and the plans' schemas.
plan_keyed() {  # plan_keyed JOB_TREE ROW_DIR: exit 0 when each plan in ROW_DIR is keyed to JOB_TREE's code, else 1 with the reason
  PYTHONPATH=$1/integrations/vllm:$1/tools/research/src "${VY_JOB_PY:-/workspace/jobs/venv312/bin/python}" - "$1" "$2" << 'EOF'
import json, os, sys
tree, row = sys.argv[1:]
try:
    from verity_vllm.pipeline import source_identity as SI
    if os.path.realpath(SI.TREE) != os.path.realpath(tree):
        sys.exit(f"source_identity came from {SI.TREE}, not the job tree {tree}")
    code = SI.code_identity(SI.TREE)["sha256"]
except Exception as e:
    sys.exit(f"the job tree's code identity isn't computable ({type(e).__name__}: {e})"[:300])
for name in ("plan/commit_plan.json", "call_boundary_plan.json"):
    if not os.path.exists(f"{row}/{name}"):
        continue
    try:
        have = (json.load(open(f"{row}/{name}")).get("key") or {}).get("code")
    except (OSError, ValueError, AttributeError) as e:
        sys.exit(f"{name} is unreadable ({type(e).__name__})")
    if have != code:
        sys.exit(f"{name} is keyed to code {str(have)[:16]}, the Commit's job tree is {code[:16]}")
EOF
}

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
  local src why
  # VY_N2_SRC: an existing job tree instead of the item tree's current one (a proof reruns a Commit from the tree it ran from)
  src=${VY_N2_SRC:-$(sudo -n -u '#1000' bash "$tree/tools/research/src/research/pods/nebius/sky/job_tree.sh" "$tree")}
  why=$(plan_keyed "$src" "$row_dir" 2>&1) \
    || { echo "$why; not queued (node 1's leased Commit derives its plans before it takes a GPU)" >&2; exit 3; }
  # staged means plain files on node 2; PoUW's /workspace/hf is read, never written
  ssh -n "$N2" "d=/workspace/jobs/hf/hub/$model/snapshots/$rev; p=/workspace/hf/hub/$model/snapshots/$rev
    if ! { [ -f \$d/config.json ] && [ -z \"\$(find \$d -type l -print -quit)\" ]; } && [ -f \$p/config.json ]; then
      mkdir -p \$d.tmp && cp -rL \$p/. \$d.tmp/ && rm -rf \$d && mv \$d.tmp \$d
    fi
    [ -f \$d/config.json ] && [ -z \"\$(find \$d -type l -print -quit)\" ]" \
    || { echo "checkpoint $repo@$rev isn't staged on node 2 (n2_build.sh stage REPO REV); not queued" >&2; exit 3; }
  local question placed
  question=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("question") or "")' "$item")
  placed=$(place commit "$key" "$(tp_of "$row")" "$(commit_max_min "$row")" "${question:-$(n2_question "$row")}") \
    || { echo "not queued" >&2; exit 3; }
  local build_run build_art
  # the Commit keeps node 1's dispatcher key, so its replay and records chain on under it
  [ -z "$job" ] || [ -n "${VY_COMMIT_KEY:-}" ] || VY_COMMIT_KEY=$(ssh -n "$N1" "$KUBE get job $job -o jsonpath='{.metadata.annotations.verity\\.dev/item}'")
  build_run=$(cat "$row_dir/two-task-build-run")
  build_art=$(cd /workspace/jobs && sudo -n -u '#1000' env UV_NO_CONFIG=1 RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=/workspace/jobs/runs \
    PYTHONPATH="$src/tools/research/src" UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python \
    /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data show "$build_run" --json 2> /dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d, list) else d; print((d.get("outputs") or {}).get("build", ""))' \
    2> /dev/null || true)
  if [ -n "$build_art" ]; then
    local m
    m=$STORE/manifests/$(art_hex "$build_art").json
    [ -s "$m" ] || { echo "Build $build_run cites $build_art, which node 1's store has no manifest for; not queued" >&2; exit 3; }
    push "$m" "$m" --ignore-existing || { echo "the manifest of $build_art didn't go over to node 2; not queued" >&2; exit 3; }
  fi
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
d["n2"]={"key":sys.argv[2],"src":sys.argv[3],"build_run":sys.argv[4],"build_art":sys.argv[5],"commit_key":sys.argv[6],"proof":sys.argv[7]=="1"}; print(json.dumps(d))' \
    "$item" "$key" "$src" "$build_run" "$build_art" "${VY_COMMIT_KEY:-n2-commit/$key}" "${VY_N2_PROOF:-0}" \
    | ssh "$N2" "mkdir -p $GUEST/items && cat > $GUEST/items/commit-$key.json"
  ssh "$N2" "f=/workspace/pouw/fill/queue/verity-commit-$key.sh; cat > \$f.tmp && chmod +x \$f.tmp && mv \$f.tmp \$f" <<EOF
#!/usr/bin/env bash
# fill: owner=$OWNER gpus=$(tp_of "$row") max_min=$(commit_max_min "$row") cpus=8 project=verity prio=$PRIO
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
  [ -z "$placed" ] || echo "$placed" >> "$GUEST/placed.jsonl"
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

# node 2's side runs in a private mount namespace: the manifests pin every checkpoint at /workspace/hf/hub/..., node 1's cache, and
# node 2's /workspace/hf is PoUW's, so the job sees /workspace/jobs/hf there (as n2_build.sh's Build). sudo drops the environment, so
# the lease's GPU and the fill runner's variables go along by name
in_ns() {  # in_ns MODE KEY
  [ -z "${VY_N2_NS:-}" ] || return 0
  exec sudo -n unshare --mount --propagation private env VY_N2_NS=1 HOME="$HOME" PATH="$PATH" USER="$USER" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
    FILL="${FILL:-}" FILL_JOB="${FILL_JOB:-}" GPU_LEASE_WHO="${GPU_LEASE_WHO:-}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    bash -c 'mount --bind /workspace/jobs/hf /workspace/hf && exec setpriv --reuid="$SUDO_UID" --regid="$SUDO_GID" --init-groups bash "$0" "$1" "$2"' \
    "$(readlink -f "$0")" "$1" "$2"
}

load_item() {  # load_item ITEM.json: config-run.yaml's task envs, then the item's over them (dispatch.py's order); HF_HOME is node 2's own
  export HF_HOME=/workspace/jobs/hf HF_HUB_OFFLINE=1 SWEEP_DIR=/workspace/cp/sweep CUDA_HOME=/workspace/jobs/cuda-12.9 \
    TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions PY=/workspace/jobs/venv312/bin/python PY312=/workspace/jobs/venv312/bin/python \
    RESEARCH_PY=/workspace/jobs/bin/py312 RESEARCH_RUNS=/workspace/jobs/runs RESEARCH_STORE=/workspace/jobs/store \
    HOT_ROOT=/workspace/jobs/hot HOT_RELEASING_MARKER=/workspace/jobs/hot/releasing HOT_RELEASE_JSON=/workspace/jobs/hot/release.json \
    GUARD_PRESERVE_FILE=/workspace/jobs/dm/guard.preserve VERITY_UNIT_RULE_CACHE=/workspace/jobs/cache/unit-rules \
    UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python CAMPAIGN=nebius-sweep
  eval "$(python3 - "$1" << 'EOF'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
env = {k: str(v) for k, v in (d.get("env") or {}).items() if k != "HF_HOME"}
n2 = d["n2"]
env.update(SRC=n2["src"], TREE=d["tree"], KEY=n2["key"], COMMIT_KEY=n2["commit_key"], BUILD_ART=n2.get("build_art") or "",
           COMMIT_ART=n2.get("commit_art") or "", REPLAY_DEFERRED="0" if n2.get("proof") else "1", PROOF="1" if n2.get("proof") else "",
           QUESTION=d.get("question") or "",
           N2_QUESTION=f"Does {env.get('ROW')}'s Commit validate on vy-nebius-2's RTX PRO 6000, its replay equal on every sampled unit,"
                       " as node 1's Commits of its Build do?")
print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()))
EOF
)"
  eval "ITEM_ARGS=($(submit_args "$1"))"
  research=(/workspace/jobs/bin/uv run --no-project --python 3.12 python -m research)
  # node 2's store has no remote, so `research run` refuses without this (an environment variable: older job trees' research has
  # no --local-only); home() is the fast path
  export RESEARCH_LOCAL_ONLY="node 2's n2-custody timer (n2_custody.sh) takes every ended run under /workspace/jobs/runs to node 1's store and R2"
  mkdir -p /workspace/jobs/hot /workspace/jobs/dm /workspace/jobs/cache/unit-rules
  [ -x /workspace/jobs/bin/py312 ] || printf '#!/bin/sh\nexec /workspace/jobs/bin/uv run --no-project --python 3.12 python "$@"\n' > /workspace/jobs/bin/py312
}

# the item's fields node 1's `dispatch.py submit` takes, as shell words: a Commit that goes back to node 1 (dispatch_n1, reclaim)
# keeps its env, class, resources, queue, priority, lease and pinned template; without its lease a `lease: self` Commit comes
# back as a whole-GPU pod that holds the GPU through its CPU prep
submit_args() {  # submit_args ITEM.json
  python3 - "$1" << 'EOF'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
args = [x for k, v in {**(d.get("env") or {}), "REPLAY_DEFERRED": "1"}.items() for x in ("--env", f"{k}={v}")]
for k, flag in (("class", "--class"), ("queue", "--queue"), ("priority", "--priority"), ("lease", "--lease"), ("template_sha", "--template-sha")):
    args += [flag, str(d[k])] if d.get(k) else []
args += ["--resources", json.dumps(d["resources"])] if d.get("resources") else []
print(shlex.join(args))
EOF
}

# the GPU's environment, for a guest Commit's record to be checked against node 1's (circuits, 2:41 PM PDT): the driver, the vLLM pin,
# the clocks and their lock state as nvidia-smi reports them while the lease holds the GPU
gpu_env() {  # gpu_env OUT.json
  local q
  q=$(nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.gr,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.max.gr,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>&1 || true)
  NV_Q=$q NV_CLOCK=$(nvidia-smi -q -d CLOCK,PERFORMANCE -i "$CUDA_VISIBLE_DEVICES" 2>&1 || true) \
    VLLM=$("$PY" -c 'import vllm; print(vllm.__version__)' 2> /dev/null || echo unknown) python3 - "$1" << 'EOF'
import json, os, socket, sys, time
f = ["index", "uuid", "name", "driver_version", "clocks_gr_mhz", "clocks_sm_mhz", "clocks_mem_mhz", "clocks_app_gr_mhz", "clocks_max_gr_mhz",
     "power_limit", "clocks_event_reasons_active"]
rows = [dict(zip(f, [x.strip() for x in line.split(",")])) for line in os.environ["NV_Q"].splitlines() if line.strip()]
json.dump({"schema": "n2-gpu-env/v0", "host": socket.gethostname(), "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": rows[0] if rows else {}, "gpus": rows,
           "vllm_version": os.environ["VLLM"],
           "nvidia_smi_clock": os.environ["NV_CLOCK"]}, open(sys.argv[1], "w"), indent=1)
EOF
}

# the row, the run dir and the Attempt back to node 1 (its store, then R2: from node 2 when node 2 has the R2 environment and the run
# dir is under 64 MiB, else from a pod on node 1)
home() {  # home RUN_ID [RSYNC-ARG...]
  local run_id=$1 sudo_rsync=(--rsync-path="sudo -n rsync" --chown=1000:1000 --mkpath)
  shift
  rsync -a "${sudo_rsync[@]}" "$@" "$SWEEP_DIR/$ROW/" "$N1:$SWEEP_DIR/$ROW/"
  rsync -a "${sudo_rsync[@]}" "/workspace/jobs/runs/$run_id/" "$N1:/workspace/jobs/runs/$run_id/"
  ssh "$N1" "$RESEARCH_N1 PYTHONPATH=$SRC/tools/research/src /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data attempt publish /workspace/jobs/runs/$run_id"
  # the Attempt's `question` labels: the research owner's (the item's "question"), then node 2's; --off-vocab for job trees older
  # than that key
  local q
  for q in "$QUESTION" "$N2_QUESTION"; do
    [ -n "$q" ] || continue
    ssh "$N1" "$RESEARCH_N1 PYTHONPATH=$SRC/tools/research/src /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data label \
      $run_id question $(printf %q "$q") --by n2-commits --ref $run_id --off-vocab" > /dev/null || echo "$run_id: no question label" >&2
  done
  if [ -n "${R2_ENDPOINT:-}" ] && [ "$(du -sm "/workspace/jobs/runs/$run_id" | cut -f1)" -lt 64 ]; then
    PYTHONPATH=$SRC/tools/research/src "${research[@]}" data push "$run_id" && echo "$run_id: pushed to R2 from node 2"
  else
    push_attempt "$run_id" "$SRC"
  fi
}

dispatch_n1() {  # dispatch_n1 TASK: node 1's dispatcher takes the item from TASK on
  [ -z "${PROOF:-}" ] || { echo "a proof Commit: nothing goes to node 1's dispatcher"; return 0; }
  # a Commit handed back goes into node 1's reclaimed.txt first, or offload moves it here again once it has waited there
  [ "$1" != 1 ] || ssh -n "$N1" "echo $(printf %q "$COMMIT_KEY") >> $GUEST/reclaimed.txt" \
    || echo "$COMMIT_KEY isn't in node 1's reclaimed.txt; offload may move it here again" >&2
  ssh "$N1" "$KUBE_DISPATCH /workspace/jobs/venv312/bin/python $DISPATCH submit config-run $COMMIT_KEY --tree $TREE --task $1 \
    $(printf '%q ' "${ITEM_ARGS[@]}")"
}

run() {
  in_ns run "$1"
  local key=$1 item=$GUEST/items/commit-$1.json run_id rc=0 research
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "no GPU leased (CUDA_VISIBLE_DEVICES is empty)" >&2; exit 2; }
  load_item "$item"
  export OV_NOISY=true
  local row_dir=$SWEEP_DIR/$ROW marker=$SWEEP_DIR/$ROW/.n2-commit
  if [ -s "$marker" ]; then
    read -r run_id rc < "$marker"
  else
    # a Commit stopped twice (its max_min, windows, waiters) goes back to node 1 rather than loop here
    local tries
    tries=$(( $(cat "$row_dir/.n2-tries" 2> /dev/null || echo 0) + 1 ))
    echo "$tries" > "$row_dir/.n2-tries"
    if [ "$tries" -gt "${VY_N2_COMMIT_TRIES:-2}" ]; then
      rsync -a --rsync-path="sudo -n rsync" --chown=1000:1000 --mkpath --exclude=/.n2-tries "$row_dir/" "$N1:$row_dir/"
      dispatch_n1 1
      rm -rf "$row_dir" "$item"
      echo "n2-commit $KEY: stopped $((tries - 1)) times on node 2; node 1 runs it"
      return 0
    fi
    fetch_manifests "$BUILD_ART"
    (cd "$SRC/integrations/vllm" && bash "$SRC/tools/research/src/research/pods/nebius/sky/vllm_bootstrap.sh" --cases "$ROLE" \
      --out "/workspace/jobs/bootstrap/$ROLE")
    export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH PYTHONPATH=$SRC/integrations/vllm:$SRC/tools/research/src
    case "$ROW" in *__bi-eager) export TRITON_CACHE_DIR=/workspace/jobs/cache/triton/$(basename "$SRC") VLLM_CACHE_ROOT=/workspace/jobs/cache/vllm/$(basename "$SRC");; esac
    if grep -q staged_sources "$SRC/integrations/vllm/verity_vllm/commit/committer/native_jit.py"; then
      export HIDDEN_GPU_BUILD=/workspace/jobs/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=/workspace/jobs/cache/jit/nc
    fi
    gpu_env "$row_dir/n2_gpu_env.json"
    cat "$row_dir/n2_gpu_env.json" | head -20
    echo "commit cites the Build: artifact ${BUILD_ART:-none}"
    local deferred=(--replay-deferred 1)
    [ -z "${PROOF:-}" ] || deferred=()
    run_id=r$(date -u +%Y%m%d-%H%M%S)-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')
    echo "$run_id" > "$row_dir/pipeline-commit-run"
    "${research[@]}" run --id "$run_id" --tool vllm.commit --project verity --cwd "$SRC" --campaign "$CAMPAIGN" \
      ${BUILD_ART:+--input "build=$BUILD_ART"} -- "$PY" -m verity_vllm.pipeline.cli row stage commit "$ROW" "$ROLE" "$REPO" "$REVISION" \
      --config-run 1 --replay-k 460 ${deferred[@]+"${deferred[@]}"} || rc=$?
    # a stop (the lease's max_min, a waiter, a window) is requeued and reruns the Commit from the top
    case "$rc" in 124 | 137 | 143) exit 143 ;; esac
    cp "$row_dir/n2_gpu_env.json" "/workspace/jobs/runs/$run_id/" 2> /dev/null || true
    echo "$run_id $rc" > "$marker"
  fi
  # a proof (VY_N2_PROOF=1 at submit) runs the Commit as node 1's gpu task did, its replay on the GPU, and only comes home: it is
  # compared with node 1's record of the same row, and nothing is queued after it. It exits with the Commit's rc, so its lease's
  # holder sees a failed proof as one
  if [ -n "${PROOF:-}" ]; then
    home "$run_id"
    rm -rf "$row_dir" "/workspace/jobs/runs/$run_id" "$item"
    echo "n2-commit $KEY: proof Commit $run_id on node 2 rc $rc; its row and Attempt are on node 1"
    return "$rc"
  fi
  if [ "$rc" != 0 ] || [ "${VY_N2_REPLAY:-n2}" = n1 ]; then
    # node 1 takes it from here: the Commit again (task 1) after a failure here, else the replay (task 2), with the whole row
    home "$run_id"
    if [ "$rc" = 0 ]; then dispatch_n1 2; else dispatch_n1 1; fi
    rm -rf "$row_dir" "/workspace/jobs/runs/$run_id" "$item"
    echo "n2-commit $KEY: Commit $run_id on node 2 rc $rc; node 1 has it now"
    return 0
  fi
  # the replay runs here, on the Verity CPUs, from the sealed bundle, which goes to node 1 only when that replay fails or leaves
  # no complete slim keep of it
  home "$run_id" --exclude='/commit/replay_bundle_p*'
  local commit_art
  commit_art=$(ssh "$N1" "$RESEARCH_N1 PYTHONPATH=$SRC/tools/research/src /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data show $run_id --json" 2> /dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d, list) else d; print((d.get("outputs") or {}).get("verdict", ""))' 2> /dev/null || true)
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["n2"]["commit_art"]=sys.argv[2]; d["n2"]["commit_run"]=sys.argv[3]; json.dump(d, open(sys.argv[1]+".tmp","w"))' \
    "$item" "$commit_art" "$run_id" && mv "$item.tmp" "$item"
  local mem
  mem=$(python3 -c 'import json,sys; m=((json.load(open(sys.argv[1])).get("resources") or {}).get("replay") or {}).get("memory") or 96; print(min(256, max(64, int(m))))' "$item")
  local f=/workspace/pouw/fill/queue/verity-replay-$key.sh
  cat > "$f.tmp" <<EOF
#!/usr/bin/env bash
# fill: owner=$OWNER gpus=0 max_min=120 cpus=8 project=verity mem_gb=$mem
# Verity vLLM replay $ROW (node 1 key $COMMIT_KEY) of the node-2 Commit $run_id: n2_commit.sh on infra/nebius.
exec bash $GUEST/bin/n2_commit.sh replay $key
EOF
  chmod +x "$f.tmp" && mv "$f.tmp" "$f"
  echo "n2-commit $KEY: Commit $run_id on node 2 rc 0 (verdict ${commit_art:-none}); its replay is queued here"
}

# a passed replay's bundle goes only with its complete slim keep, as config-run.yaml's replay task checks it (the same lines); a
# bundle kept goes home to node 1 with the row, where release.py counts it against the bundle cap
discard_kept_bundles() {  # discard_kept_bundles ROW_DIR
  python3 - "$1" << 'EOF' || echo "$(date -u +%FT%TZ) replay bundles kept: the keep check failed" | tee -a "$1/row.log" || true
import glob, hashlib, json, os, re, shutil, sys, time
row = sys.argv[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def unkept(b, k, pair):  # why k is not bundle b's complete slim keep, or None
    if not os.path.isfile(f"{k}/slim.json"):
        return f"no slim keep {k}/slim.json (a tree before keep-leaves, or its replay wrote none)"
    try:
        doc = json.load(open(f"{k}/slim.json"))
        files = doc["files"]
        if doc.get("schema") != "verity-vllm/replay-slim/v1" or doc.get("pair") != pair or not files:
            return f"{k}/slim.json is not a replay-slim/v1 keep of pair {pair} that lists files"
        if doc.get("dump_sha256") != hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            return f"{k}/slim.json's dump_sha256 is not the digest of its files"
        if doc.get("bundle_manifest_sha256") != sha256(f"{b}/MANIFEST.json"):
            return f"{k} was made from another bundle (its bundle_manifest_sha256 is not this MANIFEST.json's)"
        bad = [r for r, f in sorted(files.items()) if os.path.getsize(f"{k}/{r}") != f["bytes"] or sha256(f"{k}/{r}") != f["sha256"]]
    except Exception as e:
        return f"{k} is incomplete ({type(e).__name__}: {e})"
    return f"{len(bad)} of {k}'s {len(files)} files differ from its slim.json (first {bad[0]})" if bad else None


with open(f"{row}/row.log", "a") as log:
    for b in sorted(glob.glob(f"{row}/commit/replay_bundle_p*")):
        m = re.fullmatch(r"replay_bundle_p(\d+)", os.path.basename(b))
        k = f"{row}/commit/replay_slim_p{m[1]}" if m else ""
        why = unkept(b, k, int(m[1])) if m else "not a sealed bundle"
        if why is None:
            os.unlink(b) if os.path.islink(b) else shutil.rmtree(b)
        line = f"replay bundle {b} " + (f"kept: {why}" if why else f"deleted: {k} is its complete slim keep")
        print(line, flush=True)
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")
EOF
}

replay() {
  in_ns replay "$1"
  local key=$1 item=$GUEST/items/commit-$1.json run_id rc=0 research
  load_item "$item"
  local row_dir=$SWEEP_DIR/$ROW marker=$SWEEP_DIR/$ROW/.n2-replay
  if [ -s "$marker" ]; then
    read -r run_id rc < "$marker"
  else
    fetch_manifests "$BUILD_ART" "$COMMIT_ART"
    (cd "$SRC/integrations/vllm" && bash "$SRC/tools/research/src/research/pods/nebius/sky/vllm_bootstrap.sh" --cases "$ROLE" --cpu \
      --out "/workspace/jobs/bootstrap/$ROLE-cpu")
    export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH PYTHONPATH=$SRC/integrations/vllm:$SRC/tools/research/src CUDA_VISIBLE_DEVICES=
    export LD_LIBRARY_PATH=/workspace/jobs/cuda-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    echo "replay cites the Build ${BUILD_ART:-none} and the Commit ${COMMIT_ART:-none}"
    run_id=r$(date -u +%Y%m%d-%H%M%S)-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')
    "${research[@]}" run --id "$run_id" --tool vllm.replay --project verity --cwd "$SRC" --campaign "$CAMPAIGN" \
      ${BUILD_ART:+--input "build=$BUILD_ART"} ${COMMIT_ART:+--input "commit=$COMMIT_ART"} \
      -- "$PY" -m verity_vllm.pipeline.cli row stage replay "$ROW" "$ROLE" "$REPO" "$REVISION" --config-run 1 || rc=$?
    case "$rc" in 137 | 143) exit 143 ;; esac
    if [ "$rc" = 0 ] && [ -s "$row_dir/config_record.json" ]; then discard_kept_bundles "$row_dir"; fi
    echo "$run_id $rc" > "$marker"
  fi
  home "$run_id"
  [ "$rc" = 0 ] || dispatch_n1 2
  rm -rf "$row_dir" "/workspace/jobs/runs/$run_id" "$item"
  echo "n2-commit $KEY: replay $run_id on node 2 rc $rc"
}

# node 1's Commits that Kueue has held at least VY_OFFLOAD_WAIT_S, on a checkpoint staged on node 2, except rows the pacer keeps
# held (it deactivates every arrival, so a kept row looks like any held Commit here), one per line: JOB KEY N2KEY,
# their items in $1/N2KEY.json
held_commits() {
  KUBECONFIG=$HOME/.kube/config kubectl get jobs -l verity.dev/dispatch=node1 -o json > "$1/k8s.json"
  ssh -n "$N2" 'cd /workspace/jobs/hf/hub && for m in models--*; do for s in "$m"/snapshots/*; do [ -f "$s/config.json" ] && echo "$s"; done; done
    cd /workspace/hf/hub && for m in models--*; do for s in "$m"/snapshots/*; do [ -f "$s/config.json" ] && echo "$s"; done; done' > "$1/staged.txt" || true
  python3 - "$1" "${VY_OFFLOAD_WAIT_S:-120}" "${VY_N2_WORKSTREAMS:-vllm-epoch-run n2-build}" << 'EOF'
import calendar, json, os, re, sys, time
d, wait, streams = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
staged = set(open(f"{d}/staged.txt").read().split())
BIG = ("mistral-7b", "qwen25-7b", "qwen3-30b", "olmoe", "llama31-8b", "gemma2-9b")
def kept(row):  # rows the steward's pacer (release.py) never reactivates; Gemma-2 at batch 16+ and 1024 input tokens: top-level, 1:30 AM PDT
    m = re.search(r"__b(\d+)__", row)
    b = int(m.group(1)) if m else 0
    return b >= 64 or b >= 16 and (row.startswith(BIG) or row.startswith("gemma2-") and "__i1024__" in row)
for j in json.load(open(f"{d}/k8s.json"))["items"]:
    a, lab, st = j["metadata"]["annotations"], j["metadata"]["labels"], j.get("status") or {}
    if lab.get("verity.dev/task") != "1" or not j["spec"].get("suspend") or st.get("succeeded") or st.get("failed"):
        continue
    if time.time() - calendar.timegm(time.strptime(j["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ")) < wait:
        continue
    if a.get("verity.dev/item", "").split("/")[0] not in streams:
        continue
    item = json.loads(a["verity.dev/item-json"])
    e = item.get("env") or {}
    if item.get("template") != "config-run" or not re.search(r"__tp[12]__", e.get("ROW", "")) or kept(e.get("ROW", "")):
        continue
    if f"models--{e.get('REPO', '').replace('/', '--')}/snapshots/{e.get('REVISION')}" not in staged:
        continue
    row = os.path.join(e.get("SWEEP_DIR") or "/workspace/cp/sweep", e["ROW"])
    if not os.path.exists(f"{row}/manifest.json"):
        continue
    n2key = re.sub(r"[^a-z0-9-]+", "-", a["verity.dev/item"].lower()).strip("-")[-55:].strip("-")
    json.dump(item, open(f"{d}/{n2key}.json", "w"))
    print(j["metadata"]["name"], a["verity.dev/item"], n2key)
EOF
}

# a guest Commit that has waited VY_N2_RECLAIM_MIN (60) on node 2 (PoUW's own GPU queue refilled) goes back to node 1's queue, and
# offload leaves its key alone after that. Taking it out of node 2's queue is a mv, which fails once the runner has started it.
# The wait counts from the file's ctime, which a mv into the queue resets: a Commit released from a held dir keeps its old mtime
reclaim() {
  local f key item
  for f in $(ssh -n "$N2" "find /workspace/pouw/fill/queue -maxdepth 1 -name 'verity-commit-*.sh' -cmin +${VY_N2_RECLAIM_MIN:-60} -printf '%f\n'"); do
    key=${f#verity-commit-}; key=${key%.sh}
    ssh -n "$N2" "mkdir -p $GUEST/reclaimed && mv /workspace/pouw/fill/queue/$f $GUEST/reclaimed/$f" || continue
    item=$(mktemp)
    ssh -n "$N2" "cat $GUEST/items/commit-$key.json && rm -f $GUEST/items/commit-$key.json" > "$item"
    local commit_key tree args
    { read -r commit_key; read -r tree; } < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["n2"]["commit_key"]); print(d["tree"])' "$item")
    eval "args=($(submit_args "$item"))"
    KUBECONFIG=$HOME/.kube/config /workspace/jobs/venv312/bin/python "$DISPATCH" submit config-run "$commit_key" --tree "$tree" --task 1 "${args[@]}"
    echo "$commit_key" >> "$GUEST/reclaimed.txt"
    rm -f "$item"
    echo "reclaimed $key: waited ${VY_N2_RECLAIM_MIN:-60} min on node 2; back in node 1's queue"
  done
}

# the GPUs node 1's deployments-gpu could still admit: its unused nominal quota (a lent GPU comes back by reclaim) or the cohort's
# idle GPUs within its borrowing limit. Prints nothing and fails when Kueue can't be read.
n1_free() {
  KUBECONFIG=$HOME/.kube/config kubectl get clusterqueue -o json | python3 -c '
import json, sys
cqs, G = json.load(sys.stdin)["items"], "nvidia.com/gpu"
def gpu(cq):
    q = [r for rg in cq["spec"]["resourceGroups"] for f in rg["flavors"] for r in f["resources"] if r["name"] == G]
    used = [r for f in (cq.get("status") or {}).get("flavorsUsage") or [] for r in f["resources"] if r["name"] == G]
    bl = [float(r["borrowingLimit"]) for r in q if "borrowingLimit" in r]
    return sum(float(r.get("nominalQuota", 0)) for r in q), sum(bl) if len(bl) == len(q) else float("inf"), sum(float(r.get("total", 0)) for r in used)
me = next(c for c in cqs if c["metadata"]["name"] == sys.argv[1])
cohort = [gpu(c) for c in cqs if (c["spec"].get("cohort") or c["spec"].get("cohortName")) == (me["spec"].get("cohort") or me["spec"].get("cohortName"))]
nom, bl, use = gpu(me)
idle = sum(c[0] for c in cohort) - sum(c[2] for c in cohort)
print(int(max(0, min(nom + bl - use, max(nom - use, idle)))))
' "${VY_N1_GPU_QUEUE:-deployments-gpu}"
}

# move each held Commit on a staged checkpoint to node 2 (node 2 first: top-level, 1 Oct 12:14 AM PDT, with node 2's GPUs idle and
# its disk at 38%; VY_N2_FIRST=0 restores circuits' 30 Sep 7:24 PM rule, only while n1_free prints 0) while node 2 has fewer than
# VY_N2_MAX_COMMITS Verity Commits queued; `offload --loop` repeats every VY_OFFLOAD_EVERY_S (unit vy-n2-commit-offload on node 1,
# as research)
offload() {
  case "${1:-}" in
    --loop) while :; do offload || true; sleep "${VY_OFFLOAD_EVERY_S:-120}"; done ;;
    --list) (tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; held_commits "$tmp"); return ;;
  esac
  (
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    n2_quiet || exit 0
    reclaim
    while read -r job key n2key; do
      grep -qxF "$key" "$GUEST/reclaimed.txt" 2> /dev/null && continue
      if [ "${VY_N2_FIRST:-1}" = 0 ]; then
        free=$(n1_free) || free=
        [ "$free" = 0 ] || { echo "node 1 has ${free:-unknown (Kueue unreadable)} free GPUs in deployments-gpu; $job stays"; break; }
      fi
      queued=$(ssh -n "$N2" 'ls /workspace/pouw/fill/queue/ | grep -c "^verity-commit-"' || true)
      [ "${queued:-0}" -lt "${VY_N2_MAX_COMMITS:-16}" ] || { echo "node 2 has $queued Verity Commits waiting; $job stays"; break; }
      VY_COMMIT_KEY=$key bash "$0" submit "$n2key" "$tmp/$n2key.json" "$job" < /dev/null 2> >(grep -v '^rsync' >&2) \
        || echo "$job: not moved" >&2
    done < <(held_commits "$tmp")
  )
}

cmd=${1:-}; shift || true
case "$cmd" in
  submit) submit "$@" ;;
  item) item "$@" ;;
  run) run "$@" ;;
  replay) replay "$@" ;;
  offload) offload "$@" ;;
  n1-free) n1_free ;;
  check) check_item "$@" ;;
  place) place "$@" ;;
  max-min) commit_max_min "$@" ;;
  plan-keyed) plan_keyed "$@" ;;
  fetch-manifests) fetch_manifests "$@" ;;
  *) sed -n '7,18p' "$0" >&2; exit 2 ;;
esac
