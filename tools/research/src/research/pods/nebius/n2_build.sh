#!/usr/bin/env bash
# Verity's vLLM CPU Builds on vy-nebius-2's idle CPUs (Daniel 19:12Z: the two nodes are one pool), as `project=verity` fill guests
# that node 2's fill runner pins to its Verity CPUs and freezes (SIGSTOP) through PoUW's timed windows. The Commit stays on node 1.
#
#   on node 1:  n2_build.sh submit KEY ITEM.json   (ITEM: the config-run item a lane gives dispatch.py, as in its ready files)
#   on node 2:  n2_build.sh run KEY                (the fill job's body; submit writes it)
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
DISPATCH=${VY_DISPATCH:-/workspace/jobs/dispatch/infra/nebius/dispatch.py}

n2_quiet() { ssh -o ConnectTimeout=10 "$N2" head -1 /workspace/pouw/fill/status.txt | grep -q "timed False, window waiting False"; }

# rsync to node 2, only outside a window; a window that starts stops the transfer, which resumes after it (--partial)
push() {
  local src=$1 dst=$2 pid
  while :; do
    until n2_quiet; do sleep 30; done
    nice -n 19 ionice -c3 rsync -a --partial --mkpath --bwlimit=300m "$src" "$N2:$dst" & pid=$!
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

submit() {
  local key=$1 item=$2
  [[ $key =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || { echo "KEY: lowercase letters, digits and dashes" >&2; exit 2; }
  local tree row repo rev mem max_min
  { read -r tree; read -r row; read -r repo; read -r rev; read -r mem; read -r max_min; } < <(item_get "$item" tree ROW REPO REVISION mem max_min)
  [ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("template"))' "$item")" = config-run ] \
    || { echo "ITEM: a config-run item" >&2; exit 2; }
  # node 2 caps a guest with MemoryMax, node 1 only requests: a Build has used 2-3x its request (the steward, 18:05Z), so 3x, in 64-256
  mem=${VY_N2_MEM_GB:-$(( ${mem:-32} * 3 < 64 ? 64 : ${mem:-32} * 3 > 256 ? 256 : ${mem:-32} * 3 ))} max_min=${VY_MAX_MIN:-${max_min:-130}}
  local src model
  src=$(sudo -n -u '#1000' bash "$tree/tools/research/src/research/pods/nebius/sky/job_tree.sh" "$tree")
  model=/workspace/hf/hub/models--${repo//\//--}
  [ -d "$model/snapshots/$rev" ] || { echo "no checkpoint $model/snapshots/$rev on node 1" >&2; exit 2; }
  push "$src/" "$src/"
  push "$model/" "/workspace/jobs/hf/hub/$(basename "$model")/"
  push /workspace/jobs/bootstrap/stamps/ /workspace/jobs/bootstrap/stamps/
  push /workspace/jobs/cache/unit-rules/ /workspace/jobs/cache/unit-rules/
  push "$0" "$GUEST/bin/n2_build.sh"
  local cpus=16
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["n2"]={"key":sys.argv[2],"src":sys.argv[3],"mem_gb":int(sys.argv[4]),"cpus":int(sys.argv[5])}; print(json.dumps(d))' \
    "$item" "$key" "$src" "$mem" "$cpus" | ssh "$N2" "mkdir -p $GUEST/items && cat > $GUEST/items/$key.json"
  ssh "$N2" "f=/workspace/pouw/fill/queue/verity-build-$key.sh; cat > \$f.tmp && chmod +x \$f.tmp && mv \$f.tmp \$f" <<EOF
#!/usr/bin/env bash
# fill: owner=kueue-fold gpus=0 max_min=$max_min cpus=$cpus project=verity mem_gb=$mem
# Verity vLLM Build $row (node 1 key $key): n2_build.sh on infra/nebius; its Commit runs on node 1.
exec bash $GUEST/bin/n2_build.sh run $key
EOF
  echo "queued verity-build-$key on node 2 ($row, ${mem} GB)"
}

run() {
  local key=$1 item=$GUEST/items/$1.json
  # config-run.yaml's build envs, then the item's over them (dispatch.py's order); HF_HOME is node 2's own (see above)
  export HF_HOME=/workspace/jobs/hf HF_HUB_OFFLINE=1 SWEEP_DIR=/workspace/cp/sweep CUDA_HOME=/workspace/jobs/cuda-12.9 \
    TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions PY=/workspace/jobs/venv312/bin/python PY312=/workspace/jobs/venv312/bin/python \
    RESEARCH_PY=/workspace/jobs/bin/py312 RESEARCH_RUNS=/workspace/jobs/runs RESEARCH_STORE=/workspace/jobs/store \
    HOT_ROOT=/workspace/jobs/hot HOT_RELEASING_MARKER=/workspace/jobs/hot/releasing HOT_RELEASE_JSON=/workspace/jobs/hot/release.json \
    GUARD_PRESERVE_FILE=/workspace/jobs/dm/guard.preserve BUILD_JOBS=auto \
    VERITY_UNIT_RULE_CACHE=/workspace/jobs/cache/unit-rules UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python \
    CAMPAIGN=nebius-sweep
  eval "$(python3 - "$item" << 'EOF'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
env = {k: str(v) for k, v in (d.get("env") or {}).items() if k != "HF_HOME"}
n2 = d["n2"]
env.update(SRC=n2["src"], TREE=d["tree"], KEY=n2["key"], BUILD_RAM_BUDGET_GB=str(n2["mem_gb"]), OMP_NUM_THREADS=str(n2["cpus"]))
print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()))
print("COMMIT_ARGS=(" + " ".join(shlex.quote(a) for a in
      [x for k, v in (d.get("env") or {}).items() for x in ("--env", f"{k}={v}")]
      + ([ "--class", d["class"]] if d.get("class") else [])
      + (["--resources", json.dumps(d["resources"])] if d.get("resources") else [])
      + (["--queue", d["queue"]] if d.get("queue") else []) + (["--priority", d["priority"]] if d.get("priority") else [])) + ")")
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
  ssh "$N1" "sudo -n -u '#1000' env RESEARCH_STORE=/workspace/jobs/store RESEARCH_RUNS=/workspace/jobs/runs PYTHONPATH=$SRC/tools/research/src \
    UV_CACHE_DIR=/workspace/jobs/cache/uv UV_PYTHON_INSTALL_DIR=/workspace/jobs/cache/python \
    /workspace/jobs/bin/uv run --no-project --python 3.12 python -m research data attempt publish /workspace/jobs/runs/$run_id"
  ssh "$N1" "KUBECONFIG=\$HOME/.kube/config /workspace/jobs/venv312/bin/python $DISPATCH submit config-run n2-build/$KEY --tree $TREE --task 1 \
    $(printf '%q ' "${COMMIT_ARGS[@]}")"
  echo "n2-build $KEY: Build $run_id on node 2, Commit submitted on node 1"
}

cmd=${1:-}; shift || true
case "$cmd" in
  submit) submit "$@" ;;
  run) run "$@" ;;
  *) sed -n '2,6p' "$0" >&2; exit 2 ;;
esac
