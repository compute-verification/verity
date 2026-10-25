#!/usr/bin/env bash
# B1's 1-5 MB points on A2's GSM8K curve, one leased GPU, one SEED: for each of ARMS (default lora1 lora1h lora2), GRPO with
# A2's grpo.py and recipe (a2/, 150 steps, in-job greedy and T = 1 evals), then counted.py (watch.py) on its final checkpoint;
# then every final checkpoint served unmerged through vLLM's LoRA path (as A2 evaluated lora16), then the step-75 checkpoints.
# A phase whose estimate would run past the job's end (START + MAX_MIN) is skipped, never cut short.
#   research run --queue --source CHECKOUT --project verity --gpus 1 --max-min 90 --cwd source \
#     --send gsm8k_train.jsonl --send gsm8k_test.jsonl --send gsm8k.txt --env SEED=1 -- bash benchmarks/tvi/b1/mb.sh
set -uo pipefail
START=$(date +%s)
IN="$RESEARCH_RUN_DIR/inputs"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /workspace/jobs/tvi/scratch
S=$(mktemp -d /workspace/jobs/tvi/scratch/b1mb.XXXXXX)
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$HERE"/a2/grpo.py "$HERE"/a2/evalckpts.py "$HERE"/counted.py "$HERE"/quant.py "$HERE"/watch.py "$HERE"/ckpts.py "$S"/
mkdir -p "$S/data"
cp "$IN"/gsm8k_train.jsonl "$IN"/gsm8k_test.jsonl "$S/data/"
cp "$IN"/gsm8k.txt "$S/"
cd "$S"

SEED="${SEED:-1}" STEPS="${STEPS:-150}" ARMS="${ARMS:-lora1 lora1h lora2}"
END=$((START + ${MAX_MIN:-90} * 60 - 180))
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PYV=/workspace/jobs/venv312/bin/python
if [ -f /workspace/jobs/tvi/venv/READY ] && /workspace/jobs/tvi/venv/bin/python -c 'import vllm, transformers, torch' 2>/dev/null; then
  PYV=/workspace/jobs/tvi/venv/bin/python
fi
PYC=/workspace/jobs/venv312/bin/python
if ! "$PYV" -c 'import math_verify' 2>/dev/null; then
  uv pip install -q --python "$PYV" --target "$S/extra" --no-deps math-verify latex2sympy2_extended antlr4-python3-runtime==4.13.2 \
    && export PYTHONPATH="$S/extra"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0 MODEL DATA="$S/data" SVD_CACHE="$S/svd.pt"
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
CKPT_ROOT="/workspace/jobs/tvi/ckpt/B1/s$SEED/A2"
CLAIM="/workspace/jobs/tvi/b1/claimed-mb/${RESEARCH_RUN_ID:-$START}"
mkdir -p "$CKPT_ROOT" "$CLAIM"
if command -v research >/dev/null; then
  for p in /workspace/jobs/tvi/ckpt/B1 /workspace/jobs/tvi/b1; do
    research keep "$p" --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program' || true
  done
fi
echo "python: $PYV (train), $PYC (counted); ckpts $CKPT_ROOT; seed $SEED; arms $ARMS; end $(date -u -d @"$END" +%H:%MZ)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"$PYV" -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__, torch.cuda.get_device_name(0))'
(while :; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed "s/^/$(date -u +%H:%M:%S) /"; sleep 60; done) \
  > "$OUT/gpu.log" 2>&1 &
MON=$!

fits() {  # fits EST_MIN NAME: the phase ends before END, else it is skipped and said so
  if [ $(( $(date +%s) + $1 * 60 )) -gt "$END" ]; then
    echo "$(date -u +%H:%MZ) skip $2: ~$1 min would run past $(date -u -d @"$END" +%H:%MZ)"
    return 1
  fi
}

analyze() {  # analyze STEPGLOB EST_MIN: counted.py on this seed's unclaimed checkpoints matching STEPGLOB
  fits "$2" "counted $1" || return 0
  OUT="$OUT" CLAIM="$CLAIM" WATCH_ROOTS="$CKPT_ROOT" WATCH_STEPS="$1" TEXT_OVERRIDE=gsm8k.txt TEXTS="$S" IDLE_MIN=0 \
    DEADLINE="$(date -u -d @"$END" +%Y-%m-%dT%H:%M:%SZ)" PYTHONPATH="$S" \
    timeout $(( END - $(date +%s) )) "$PYC" watch.py 2>&1 \
    | grep --line-buffered -v -E 'Loading weights|unauthenticated|Writing model' >> "$OUT/watch.log"
  echo "$(date -u +%H:%MZ) counted $1 done"
  grep -E '^A2_' "$OUT/watch.log" | tail -1
}

for arm in $ARMS; do
  fits "${ARM_MIN:-19}" "$arm" || continue
  timeout $(( END - $(date +%s) )) "$PYV" grpo.py --task gsm8k --method "$arm" --seed "$SEED" --steps "$STEPS" --model "$MODEL" \
    --data "$S/data" --out "$OUT" --ckpt-root "$CKPT_ROOT" --deadline "$END" > "$OUT/log-gsm8k-$arm-seed$SEED.txt" 2>&1
  echo "$(date -u +%H:%MZ) gsm8k-$arm-seed$SEED rc=$?"
  grep -h '^{' "$OUT/log-gsm8k-$arm-seed$SEED.txt" | grep -E '"(setup|done|final_sampled|deadline_stop)"' || true
  analyze "step$STEPS" 4
done

if fits 8 "unmerged evals"; then
  timeout $(( END - $(date +%s) )) "$PYV" evalckpts.py --root "$CKPT_ROOT" --glob "*/step$STEPS" --model "$MODEL" --data "$S/data" \
    --out "$OUT" --n 4 --temperature 1.0 --unmerged "${ARMS// /+}" > "$OUT/log-evalckpts.txt" 2>&1
  echo "$(date -u +%H:%MZ) unmerged evals rc=$?"
  grep -h '^{' "$OUT/log-evalckpts.txt" || true
fi
analyze "step*" 8
kill "$MON" 2>/dev/null
ls -la "$OUT"
