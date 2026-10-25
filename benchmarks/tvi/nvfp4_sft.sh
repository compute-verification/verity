# The NVFP4 training study on one leased GPU: nvfp4_sft.py's arms one after another, one ITEMS entry
# `ARM:ORDER:LR:SEED:MAX_STEPS` each (comma-separated). An item stops at val exact >= 0.995, MAX_STEPS, ITEM_MAX_MIN minutes
# or DEADLINE_EPOCH, then runs its test eval; an item that would start within 5 min of DEADLINE_EPOCH is skipped.
# Writes $OUT/<arm>_<task>_lr<lr>_s<seed>/result.json, its log, and $OUT/table.md.
set -u
D=$(cd "$(dirname "$0")" && pwd)
SCR=$(mktemp -d "${SCRATCH_ROOT:-/tmp}/nvfp4_sft.XXXXXX")
trap 'rm -rf "$SCR"' EXIT
trap 'exit 143' TERM INT
M=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DEADLINE_EPOCH=${DEADLINE_EPOCH:-$(( $(date +%s) + 80 * 60 ))}
ITEM_MAX_MIN=${ITEM_MAX_MIN:-30}

PY=${TVI_PY:-}
[ -n "$PY" ] || for c in /workspace/jobs/tvi/venv/bin/python /workspace/jobs/venv312/bin/python; do
  if [ -x "$c" ] && "$c" -c 'import torch, transformers; assert torch.cuda.is_available()' 2>/dev/null; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  uv venv -q -p 3.12 "$SCR/venv"
  uv pip install -q --python "$SCR/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python "$SCR/venv/bin/python" transformers safetensors
  PY="$SCR/venv/bin/python"
fi
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$SCR/hf"; fi
if [ -d "${HF_HUB_CACHE:-/nonexistent}/models--${M//\//--}" ]; then export HF_HUB_OFFLINE=1; fi
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "PY=$PY $("$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__)')"
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} deadline=$(date -u -d @"$DEADLINE_EPOCH" +%TZ) items=$ITEMS"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

IFS=, read -ra LIST <<< "$ITEMS"
for item in "${LIST[@]}"; do
  IFS=: read -r arm order lr seed steps <<< "$item"
  now=$(date +%s)
  if [ $((now + 300)) -gt "$DEADLINE_EPOCH" ]; then echo "skip $item: deadline in $((DEADLINE_EPOCH - now))s"; continue; fi
  dl=$(( now + ITEM_MAX_MIN * 60 < DEADLINE_EPOCH - 180 ? now + ITEM_MAX_MIN * 60 : DEADLINE_EPOCH - 180 ))
  n=${arm}_o${order}_lr${lr}_s${seed}
  echo "== $n $(date -u +%TZ)"
  "$PY" "$D/nvfp4_sft.py" train --arm "$arm" --order "$order" --lr "$lr" --seed "$seed" --max-steps "$steps" --model "$M" \
    --deadline-epoch "$dl" --out "$OUT/$n" ${TRAIN_ARGS:-} > "$OUT/$n.log" 2>&1
  echo "$n rc=$? in $(( $(date +%s) - now ))s: $(tail -1 "$OUT/$n.log")"
done
"$PY" "$D/nvfp4_sft.py" table "$OUT" > "$OUT/table.md" 2>&1
cat "$OUT/table.md"
