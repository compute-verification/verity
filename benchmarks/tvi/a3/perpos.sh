#!/usr/bin/env bash
# Per-position accuracy and learned key bits of A3's checkpoints (GLOB under the ckpt root), on the leased GPU.
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
S=$(mktemp -d "${SCRATCH_ROOT:-/tmp}/a3pp.XXXXXX")
trap 'rm -rf "$S"' EXIT
cp "$IN"/newskill.py "$IN"/perpos.py "$S"/
cd "$S"
PY=
for c in /workspace/jobs/tvi/venv/bin/python /workspace/jobs/venv312/bin/python /workspace/jobs/tvi/venv-A2/bin/python; do
  if [ -x "$c" ] && "$c" -c 'import torch, transformers' 2>/dev/null; then PY=$c; break; fi
done
export HF_HUB_CACHE=/workspace/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" perpos.py --glob "${GLOB:-*-seed0}" --out "$OUT/perpos.json"
