#!/usr/bin/env bash
# β(C) (beta.py) on the exact-int8 Llama-3.1-8B request Program (job.sh's Build, digest 8c683b2a..., LP=256, T=31), with the
# input term shared over the 8B row's KV pool (share.py's M_dyn for 8 engines). ART=art:... reads another Build's tree,
# M_NODE=BYTES another pool, CHUNKS="N ..." the coarse bounds before the fine one, PROBE=1 stops once the Program decodes.
# Where /workspace/jobs/venv312 isn't (the CPU nodes), it syncs the workspace with CPU torch into scratch. A node with no store
# remote takes the descriptor sent along (`research data fetch ART --path descriptor.json.gz`, then --send it). A descriptor
# whose primitives' signatures have since changed under the same id decodes only against its own registry: send that commit's
# packages, `git archive C verity integrations/vllm/verity_vllm | gzip > tree.tar.gz`, and they go ahead on PYTHONPATH (the
# 8B Build decodes at a4f5a429c5, the census's commit: LutIndex_v1 has returned 8 bits since, not 32).
#
#   research run --on vy-nebius-cpu-5 --timeout 5h --source . --project verity --send DIR/descriptor.json.gz \
#     --send DIR/tree.tar.gz --cwd clone/integrations/vllm -- bash ../../benchmarks/stacked_share/beta_job.sh
set -euo pipefail
export HF_HUB_OFFLINE=1
HERE=../../benchmarks/stacked_share
ART=${ART:-art:c2086c4c7da0041dcbf0776357bce6c3d7127bef6d6245c8652a772024af240f}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
if [ -x /workspace/jobs/venv312/bin/python3 ]; then
  PY=/workspace/jobs/venv312/bin/python3
  export PYTHONPATH=.:../..:../../tools/research/src
else
  UV_PROJECT_ENVIRONMENT="$d/venv" uv sync -q --all-packages --extra torch-cpu > "$OUT/sync.log" 2>&1 || { tail -30 "$OUT/sync.log"; exit 4; }
  PY=$d/venv/bin/python
fi
tree=${RESEARCH_RUN_DIR:-/nonexistent}/inputs/${TREE:-tree.tar.gz}
if [ -f "$tree" ]; then
  mkdir "$d/tree" && tar -xzf "$tree" -C "$d/tree"
  export PYTHONPATH=$d/tree:$d/tree/integrations/vllm${PYTHONPATH:+:$PYTHONPATH}
fi
desc=${RESEARCH_RUN_DIR:-/nonexistent}/inputs/descriptor.json.gz
if [ ! -f "$desc" ]; then
  "$PY" -m research data fetch "$ART" --to "$d/build" --path descriptor.json.gz > "$OUT/fetch.log" 2>&1 || { cat "$OUT/fetch.log"; exit 3; }
  desc=$(find "$d/build" -name descriptor.json.gz | head -1)
  [ -n "$desc" ] || { cat "$OUT/fetch.log"; exit 3; }
fi
cd "$d"  # the clone's integrations/vllm is no longer the cwd, so a sent tree's verity_vllm is the one imported
if [ -n "${PROBE:-}" ]; then
  nproc; free -g | head -2
  "$PY" -c "import sys; sys.path.insert(0, '$HERE'); from units import load; p = load(__import__('pathlib').Path('$desc')); \
print('root nodes', len(p.root.body.nodes))"
  exit 0
fi
chunks=()
for n in ${CHUNKS:-4096 256}; do chunks+=(--chunk "$n"); done
/usr/bin/time -v "$PY" "$HERE/beta.py" "$desc" --out "$OUT/beta-8b.json" --m-dyn-node "${M_NODE:-609801469952}" "${chunks[@]}" \
  2> >(tee "$OUT/beta.log" >&2)
