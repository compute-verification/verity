#!/usr/bin/env bash
# B3 synthetic facts on one leased GPU: run PLAN's tasks (driver.py), then tabulate. Results in $OUT.
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
S=$(mktemp -d "${SCRATCH_ROOT:-/tmp}/b3.XXXXXX")
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$IN"/*.py "$IN"/*.json "$S"/
cd "$S"
export OMP_NUM_THREADS="${THREADS:-2}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=/workspace/jobs/venv312/bin/python
if ! [ -x "$PY" ]; then
  if [ -f /workspace/jobs/tvi/venv/READY ] && /workspace/jobs/tvi/venv/bin/python -c 'import torch, transformers'; then
    PY=/workspace/jobs/tvi/venv/bin/python
  else
    uv venv -q -p 3.12 "$S/venv"
    uv pip install -q --python "$S/venv/bin/python" torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
    uv pip install -q --python "$S/venv/bin/python" transformers==5.17.0 numpy
    PY="$S/venv/bin/python"
  fi
fi
export HF_HUB_CACHE=/workspace/hf/hub HF_HUB_OFFLINE=1
if [ "${CKPT_ROOT:-}" = scratch ]; then export CKPT_ROOT="$S/ckpt"; fi
if [ -n "${CKPT_ROOT:-}" ]; then
  mkdir -p "$CKPT_ROOT"
  if [[ "$CKPT_ROOT" != /workspace/jobs/tvi/* ]]; then
    :
  elif command -v research >/dev/null; then
    research keep /workspace/jobs/tvi/ckpt/B3 --owner @training --expires 2026-10-27 --priority normal \
      --why 'training-vs-inference program' || echo "research keep failed"
  else
    echo "research not on PATH: keep /workspace/jobs/tvi/ckpt/B3 from a shell on the node"
  fi
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'
"$PY" facts.py > "$OUT/facts.txt"
cp "${PLAN:-plan.json}" "$OUT/plan.json"
"$PY" driver.py "${PLAN:-plan.json}" 2>&1 | tee "$OUT/driver.log"
"$PY" table.py "$OUT" | tee "$OUT/table.md"
