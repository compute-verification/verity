#!/usr/bin/env bash
# B2 decoding stress test on one GPU: for each model, train main + proxies (fp32 snapshots in scratch), run the
# predictors against main's steps 1, 10, 100, 1000, delete the snapshots.
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
ROOT=/tmp
if [ "$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)" -lt 80 ]; then
  ROOT=/workspace/jobs/tvi
  mkdir -p "$ROOT"
fi
S=$(mktemp -d "$ROOT/b2.XXXXXX")
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
if [ "$ROOT" != /tmp ] && command -v research >/dev/null; then
  research keep "$S" --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program' || true
fi
export SCRATCH="$S"
cp "$IN"/*.py "$S"/
cd "$S"

PY=/workspace/jobs/venv312/bin/python
if ! { [ -x "$PY" ] && "$PY" -c 'import torch, transformers' 2>/dev/null; }; then
  uv venv -q -p 3.12 "$S/venv"
  uv pip install -q --python "$S/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python "$S/venv/bin/python" transformers numpy accelerate
  PY="$S/venv/bin/python"
fi
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi

nvidia-smi --query-gpu=name,memory.total --format=csv
df -h "$S"
"$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'

for MODEL in ${MODELS:-EleutherAI/pythia-160m}; do
  export MODEL
  TAG="${MODEL##*/}"
  "$PY" train.py tok
  for cond in main b1 b1lr sub10 sub10lr1 sub10lr3; do
    "$PY" train.py "$cond" 2>&1 | sed -u '/Loading weights\|Warning/d' | tee "$OUT/train_${TAG}_${cond}.log"
  done
  case "$MODEL" in Qwen/*) export B2_LAYERS="${QWEN_LAYERS:-2 7 12 17 22}" ;; *) unset B2_LAYERS ;; esac
  "$PY" predict.py 2>&1 | sed -u '/Loading weights\|Warning/d' | tee "$OUT/predict_${TAG}.log"
  rm -f snap_"$TAG"_*.pt
done
ls -la "$OUT"
