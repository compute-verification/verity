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
export HF_HUB_CACHE=/workspace/hf/hub HF_HUB_OFFLINE=1
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
"$PY" facts.py | head -4 | tee "$OUT/facts.txt"
cp "${PLAN:-plan.json}" "$OUT/plan.json"
"$PY" driver.py "${PLAN:-plan.json}" 2>&1 | tee "$OUT/driver.log"
"$PY" table.py "$OUT" | tee "$OUT/table.md"
