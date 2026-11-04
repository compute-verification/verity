#!/usr/bin/env bash
# The scorer's refresh as a research run on a CPU node (the agent VM only edits); node_loop.py launches it:
#   research run --on vy-nebius-cpu-9 --pool research --cwd source --store-read [--send served.json --send vm-put.log]
#       -- bash benchmarks/covert_witness/scorer/refresh-node.sh
# Its state is kept on the node in K ($TE_NODE_STATE): repo.git, the bare repository node_loop.py pushes the VM's refs
# into (the records' commits, origin's branches); wt/, its worktrees (TE40_WT, the re-score's, the ingest's and the pin's
# with files, the capture layers without); state/, ledger/ and v1/, the scorer's state, its ledger and the refreshes;
# traces/, the shards tracing beside a refresh; venv/ and trace-venv/ (torch, as job40.sh's).
# MODE=bootstrap, with --send state.tar (state/, ledger/, v1/) and TE_WORKTREES (NAME:COMMIT:files|nofiles,...), builds
# K once, after node_loop.py's push. A refresh runs the ledger pass (collect, score, the search's efficiency table), then
# v1cycle.py, whose tables and frontier-branches go to out/ (custody stores them), and leaves in out/vm/ what the VM
# reads: cycle.json, digest.txt, post.json, the console bodies it PUTs (put/), the panels, collect.json (its
# commits_missing, which the VM pushes) and merge.bundle when the re-score commit is on no origin branch.
set -euo pipefail
K=${TE_NODE_STATE:-/workspace/training/te-scorer}
SRC=$(pwd)
OUT=${OUT:-$RESEARCH_RUN_DIR/out}
IN=$RESEARCH_RUN_DIR/inputs
SC=$SRC/benchmarks/covert_witness/scorer
R=$K/repo.git
mkdir -p "$OUT/vm"
say() { echo "$(date -u +%H:%M:%SZ) $*"; }

if [ "${MODE:-refresh}" = bootstrap ]; then
  [ -d "$R" ] || { echo "refresh-node: no $R (node_loop.py init creates it and pushes)" >&2; exit 2; }
  if [ -f "$R/objects/info/alternates" ]; then
    # the push borrowed research's own repository's objects: copied in, so its eviction can't take them
    say "repack"
    git -C "$R" repack -a -d -q
    rm "$R/objects/info/alternates"
  fi
  git -C "$R" config user.name "$TE_GIT_NAME"
  git -C "$R" config user.email "$TE_GIT_EMAIL"
  say "state"
  tar -xf "$IN/state.tar" -C "$K"
  mkdir -p "$K/wt" "$K/traces"
  IFS=, read -ra wts <<< "$TE_WORKTREES"
  for w in "${wts[@]}"; do
    IFS=: read -r name commit how <<< "$w"
    [ -e "$K/wt/$name" ] && continue
    git -C "$R" worktree add -q $([ "$how" = files ] || echo --no-checkout) --detach "$K/wt/$name" "$commit"
  done
  say "venvs"
  [ -x "$K/venv/bin/python" ] || { uv venv -q --python 3.12 "$K/venv"
    uv pip install -q --python "$K/venv/bin/python" "numpy==2.5.3" "matplotlib==3.11.2"; }
  if [ ! -x "$K/trace-venv/bin/python" ]; then
    uv venv -q --python 3.12 "$K/trace-venv"
    uv pip install -q --python "$K/trace-venv/bin/python" --index-strategy unsafe-best-match \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      "torch==2.15.0+cpu" "numpy==2.5.4" "safetensors==0.8.0" "tiktoken==0.14.0"
  fi
  research keep "$K" --owner @training --expires never --priority keep \
    --why "the scorer's refresh state on this node (benchmarks/covert_witness/scorer/refresh-node.sh)"
  touch "$K/.ready"
  du -sh "$K"/* | tee "$OUT/vm/bootstrap-du.txt"
  exit 0
fi

[ -f "$K/.ready" ] || { echo "refresh-node: $K isn't bootstrapped (MODE=bootstrap)" >&2; exit 2; }
export TE_SCORER_STATE=$K/state TE_SCORER=$K/state TE_LEDGER=$K/ledger TE_LEDGER_40=$K/ledger-40 TE_REPO=$R \
  TE40_WT=$K/wt/te40 TE_RESCORE_WT=$K/wt/rescore TE_NEXT_WT=$K/wt/te-next TE_CURRENT=$K/wt/te-core \
  TE_PIN=$K/wt/te-core-pin TE_OFFICIAL_SRC=$K/wt/official TE_TORCH_PYTHON=$K/trace-venv/bin/python \
  TE_TRACE_VENV=$K/trace-venv TE_OUT_ROOT=$K/v1 TE_TRACE_HERE=$K/traces TE_PANELS_DIR=$OUT/vm/panels \
  TE_CURRENT_COMMIT=${TE_CURRENT_COMMIT:-df4acd427d3151ec8d1c955f9b3e93ff38de40f4} \
  TE_SCORER_WORKERS=${TE_SCORER_WORKERS:-8} AGENT_GUARD_LOCAL_OK=1
[ -f "$IN/served.json" ] && export TE_SERVED=$IN/served.json
# the VM's PUT of the last refresh, which retrace_order reads beside its algorithm-policy
if [ -n "${TE_VM_PUT_FOR:-}" ] && [ -f "$IN/vm-put.log" ] && [ -d "$K/v1/$TE_VM_PUT_FOR/algorithm-policy" ]; then
  cp "$IN/vm-put.log" "$K/v1/$TE_VM_PUT_FOR/algorithm-policy/vm-put.log"
fi
PY=$K/venv/bin/python
RP=$SRC/tools/research/src
cd "$SC"
say "collect"
PYTHONPATH=$RP $PY scorer.py collect > "$OUT/vm/collect.out" 2>&1 || say "collect failed: $(tail -1 "$OUT/vm/collect.out")"
cp "$K/state/collect.json" "$OUT/vm/collect.json" 2>/dev/null || true
if PYTHONPATH=$RP $PY scorer.py score > /dev/null 2> "$OUT/vm/score.err"; then
  PYTHONPATH=$RP $PY search_efficiency.py "$OUT/vm/te-night2-efficiency.md" > "$OUT/vm/efficiency.out" 2>&1 \
    || say "search_efficiency failed"
else
  say "score failed: $(tail -1 "$OUT/vm/score.err")"
fi
STAMP=$(date -u +%Y-%m-%dT%H%M%SZ)
say "refresh $STAMP"
rc=0
$PY v1cycle.py --out "$K/v1/$STAMP" --no-post ${TE_V1_ARGS:-} > "$OUT/vm/v1cycle.log" 2>&1 || rc=$?
say "v1cycle rc $rc"
D=$K/v1/$STAMP
echo "$STAMP" > "$OUT/vm/stamp"
for f in cycle.json digest.txt post.json; do [ -f "$D/$f" ] && cp "$D/$f" "$OUT/vm/"; done
if [ -d "$D/algorithm-policy/console" ]; then
  mkdir -p "$OUT/vm/put"
  cp "$D"/algorithm-policy/console/*.json "$OUT/vm/put/"
fi
C=$(git -C "$TE_RESCORE_WT" rev-parse HEAD)
if [ -z "$(git -C "$R" branch -r --contains "$C")" ]; then
  # rescore_head's merge, which the node can't push: the VM pushes it to RESCORE_AHEAD
  git -C "$R" update-ref refs/te-scorer/merge "$C"
  git -C "$R" bundle create -q "$OUT/vm/merge.bundle" refs/te-scorer/merge \
    --not $(git -C "$R" for-each-ref --format='%(refname)' \
            "refs/remotes/origin/${TE_RESCORE_AHEAD:-cursor/te-delta-t-40-516a}" refs/remotes/origin/cursor/covert-witness-4207)
fi
du -sh "$K"/* > "$OUT/vm/du.txt" 2>&1 || true
exit "$rc"
