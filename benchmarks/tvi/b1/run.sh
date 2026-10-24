#!/usr/bin/env bash
# B1 on one GPU. Mode sweep (default): for each of MODELS, retrain console's conditions (train.py) and run counted.py on the
# snapshots SNAPS keeps, deleting them after. Mode ckpt: run counted.py on the checkpoint dirs listed in CKPTS (one per line,
# "DIR NAME"), against BASE, on the eval text TEXT (or the stdlib held-out tokens when unset).
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
mkdir -p /workspace/jobs/tvi/scratch
S=$(mktemp -d /workspace/jobs/tvi/scratch/b1.XXXXXX)
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
export SCRATCH="$S"
cp "$IN"/*.py "$S"/
cd "$S"

PY=/workspace/jobs/venv312/bin/python
uv pip install -q --python "$PY" --target "$S/extra" --no-deps peft accelerate
export PYTHONPATH="$S/extra:$S"
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
quiet() { grep --line-buffered -v -E 'Loading weights|unauthenticated|Writing model' || true; }

if [ "${MODE:-sweep}" = ckpt ]; then
  args=()
  while read -r dir name; do [ -n "$dir" ] && args+=("$dir" "$name"); done <<< "$CKPTS"
  if [ -n "${TEXT:-}" ]; then
    export TOKENS="$S/eval.txt" TOKENIZER="$BASE"
    printf '%b' "$TEXT" > "$TOKENS"
  else
    export MODEL="$BASE"; "$PY" train.py tok; export TOKENS="$S/tokens_${BASE##*/}.pt"
  fi
  "$PY" counted.py "$BASE" "${args[@]}" 2>&1 | quiet | tee -a "$OUT/counted_ckpt.log"
  exit 0
fi

for MODEL in ${MODELS:-EleutherAI/pythia-160m}; do
  export MODEL
  TAG="${MODEL##*/}"
  "$PY" train.py tok
  for cond in ${CONDS:-ft5e-5 ft1e-5 lora ft3e-4}; do
    "$PY" train.py "$cond" 2>&1 | quiet > "$OUT/train_${TAG}_${cond}.log"
    args=()
    for f in snap_"$TAG"_"$cond"_s*.pt; do
      n="${f#snap_}"; n="${n%.pt}"; args+=("$S/$f" "$n")
    done
    TOKENS="$S/tokens_${TAG}.pt" "$PY" counted.py "$MODEL" "${args[@]}" 2>&1 | quiet | tee -a "$OUT/counted_${TAG}.log"
    rm -f snap_"$TAG"_"$cond"_s*.pt
  done
done
ls -la "$OUT"
