#!/usr/bin/env bash
# β(C) (beta.py) on the exact-int8 Llama-3.1-8B request Program the census read (job.sh's Build at
# a4f5a429c5, LP=256, T=31, 724,643 root nodes), with the input term shared over the 8B row's KV pool (share.py's M_dyn for 8
# engines). ART=art:... and ART_PATH read another Build, M_NODE=BYTES another pool, CHUNKS="N ..." the coarse bounds before the
# fine one, PROBE=1 stops once the Program decodes. The Build binds its weights as run inputs, so a root input past
# REGISTERED_ABOVE bits (default 32768: the prompt is 8,192, the smallest weight, a norm's, 65,536) is taken as registered. Where /workspace/jobs/venv312 isn't (the CPU nodes), it syncs the workspace
# with CPU torch into scratch. A node with no store remote takes the descriptor from the run's inputs/ (ART's
# out/descriptor.json.gz, fetched from the store). A descriptor whose primitives' signatures have since changed under the same id
# decodes only against its own registry: send that commit's packages, `git archive C verity integrations/vllm/verity_vllm |
# gzip > tree.tar.gz`, and they go ahead on PYTHONPATH (this Build decodes at a4f5a429c5: LutIndex_v1 has returned 8 bits
# since, not 32).
#
#   (cd integrations/vllm && RESEARCH_RUN_DIR=RUN OUT=RUN/out bash ../../benchmarks/stacked_share/beta_job.sh)
#   # on a CPU node, from a clone's root, RUN/inputs/ holding descriptor.json.gz and tree.tar.gz
set -euo pipefail
export HF_HUB_OFFLINE=1
ROOT=$(cd ../.. && pwd)
HERE=$ROOT/benchmarks/stacked_share
ART=${ART:-art:e268a20b5c85a225f3e16281979e7c2ecfcc05eb79f36455b8e50e00a0e71aa8}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
if [ -x /workspace/jobs/venv312/bin/python3 ]; then
  PY=/workspace/jobs/venv312/bin/python3
  export PYTHONPATH=$ROOT/integrations/vllm:$ROOT:$ROOT/tools/research/src
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
  "$PY" -m research data fetch "$ART" --to "$d/build" --path "${ART_PATH:-out/descriptor.json.gz}" > "$OUT/fetch.log" 2>&1 \
    || { cat "$OUT/fetch.log"; exit 3; }
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
  --registered-above "${REGISTERED_ABOVE:-32768}" 2> >(tee "$OUT/beta.log" >&2)
