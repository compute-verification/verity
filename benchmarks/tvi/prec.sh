# The precision arm: a chain of GRPO runs (Qwen2.5-0.5B-Instruct, GSM8K, full fine-tuning) on one leased GPU, one per
# ARMS item `RECIPE:SEED` (precision.py's recipes), each followed by a GSM8K-test eval of the trained weights (greedy on all
# 1319 problems, and 4 samples at T=1). Items `base` (the untrained model's eval) and `smoke_tl` (TinyLoRA under vLLM) too.
# A run that wouldn't end by DEADLINE_EPOCH is skipped. Writes $OUT/<item>/ (metrics.json, eval/eval.json) and logs.
set -u
V=${TVI_PY:-/workspace/jobs/tvi/venv/bin/python}
S=benchmarks/tvi
M=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
STEPS=${STEPS:-150}
# 5e-6 is unstable at this batch (8 prompts x 8): bf16's training reward fell from 0.54 to 0.22 in 150 steps.
LR=${LR:-1e-6}
# vLLM's warmup samples top-k/top-p through flashinfer, which JIT-builds with ninja and nvcc; neither is on PATH here.
export PATH="$(dirname "$V"):$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
CK=/workspace/jobs/tvi/ckpt/prec/$RESEARCH_RUN_ID
# seconds an item needs: setup and the eval (~6 min) plus STEPS at the recipe's s/step measured on n2 (emulation is slow)
need_s() {
  case $1 in base) echo 300; return ;; smoke_tl) echo 600; return ;; esac
  local spi
  case ${1%:*} in bf16|none) spi=2.8 ;; store_*) spi=3.9 ;; fp8) spi=6.3 ;; mxfp8) spi=5.6 ;; nvfp4) spi=13 ;; *) spi=8 ;; esac
  awk -v s="$STEPS" -v p="$spi" 'BEGIN { print int(360 + s * p) }'
}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} host=$(hostname) deadline=$(date -u -d @"$DEADLINE_EPOCH" +%TZ) arms=$ARMS"
evalm() {
  "$V" $S/eval_pass_at_k.py --model "$1" --dataset gsm8k --n-problems 0 --samples 4 --k 1 --greedy --max-tokens 1024 \
    --gpu-mem 0.5 --out "$2" > "$2.log" 2>&1
  echo "eval rc=$? $(tail -c 400 "$2/eval.json" 2>/dev/null | tr -d '\n' | cut -c1-200)"
}
for item in $ARMS; do
  now=$(date +%s)
  need=$(need_s "$item")
  if [ $((now + need)) -gt "$DEADLINE_EPOCH" ]; then echo "skip $item: needs ~${need}s, deadline in $((DEADLINE_EPOCH - now))s"; continue; fi
  t0=$now
  echo "== $item $(date -u +%TZ)"
  case $item in
    base)
      mkdir -p "$OUT/base"; evalm "$M" "$OUT/base/eval" ;;
    smoke_tl)
      mkdir -p "$OUT/smoke_tl"
      OUT=$OUT/smoke_tl SMOKE_TESTS="grpo_vllm_tinylora_hot grpo_vllm_tinylora" bash $S/smoke.sh > "$OUT/smoke_tl.log" 2>&1
      tail -4 "$OUT/smoke_tl.log" ;;
    *)
      r=${item%:*}; s=${item#*:}; n=${r}_s$s; d=$CK/$n
      "$V" $S/grpo_train.py --model "$M" --steps "$STEPS" --adapter full --precision "$r" --seed "$s" --lr "$LR" \
        --save-dir "$d" --out "$OUT/$n" > "$OUT/$n.log" 2>&1
      rc=$?
      echo "$n train rc=$rc $(grep -o '"steps_per_s": [0-9.]*' "$OUT/$n/metrics.json" 2>/dev/null)"
      [ $rc -eq 0 ] && evalm "$d" "$OUT/$n/eval" ;;
  esac
  echo "-- $item done in $(( $(date +%s) - t0 ))s"
done
