# R2, elicitation controls: GRPO on GSM8K at S = 10 kB (TinyLoRA, 2500 fp32 numbers) with the true, a random and a
# format-only reward, plus full fine-tuning as the 100% anchor, on Qwen2.5-1.5B-Instruct (I) and Qwen2.5-1.5B (B).
#
# Launched under `gpu-lease 1` with --cwd source. CONFIGS is a space-separated list of MODEL:ARM:REWARD:SEED, ARM one of
# base (evaluate only), tiny or full; each writes $OUT/<config>/{train/metrics.json,eval_gsm8k/eval.json} and its log.
set -u
V=${TVI_PY:-/workspace/jobs/tvi/r2-venv/bin/python}
S=benchmarks/tvi
: "${CONFIGS:?set CONFIGS=MODEL:ARM:REWARD:SEED ...}"
STEPS=${STEPS:-100}
FULL_STEPS=${FULL_STEPS:-$STEPS}
TINY_LR=${TINY_LR:-5e-3}
FULL_LR=${FULL_LR:-1e-6}
EVAL_N=${EVAL_N:-0}
EVAL_SETS=${EVAL_SETS:-gsm8k}
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-5400}
# 10 kB = 2500 fp32 numbers: a rank-10 LoRA-XS core per map, a combination of 100 seeded r x r projections, tied across
# the maps of each of 25 layer groups.
TINY="--adapter tinylora --tiny-rank 10 --tiny-u 100 --tiny-groups 25"
GRPO="--gen vllm --num-generations 8 --prompts-per-step 8 --micro-batch 8 --max-completion-length 512 --vllm-mem 0.25"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname) py=$V"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.used,memory.total --format=csv
CK=$(mktemp -d -p /workspace/jobs/tvi .r2-ck.XXXXXX)
trap 'rm -rf "$CK"' EXIT

phase() {
  local name=$1 log=$2; shift 2
  echo "== $name $(date -u +%FT%TZ)" | tee -a "$OUT/phases.txt"
  local t=$(date +%s)
  timeout "$TRAIN_TIMEOUT" "$V" "$@" > "$log" 2>&1
  local rc=$?
  echo "$name rc=$rc wall=$(( $(date +%s) - t ))s end=$(date -u +%FT%TZ)" | tee -a "$OUT/phases.txt"
  tail -15 "$log"
  return $rc
}

evaluate() {
  local name=$1 model=$2
  for ds in $EVAL_SETS; do
    phase "$name/eval_$ds" "$OUT/$name/eval_$ds.log" $S/eval_pass_at_k.py --model "$model" --dataset "$ds" \
      --n-problems "$EVAL_N" --samples 4 --k 1,4 --greedy --temperature 1.0 --max-tokens 1024 --seed 0 --gpu-mem 0.85 \
      --out "$OUT/$name/eval_$ds"
  done
}

for cfg in $CONFIGS; do
  IFS=: read -r m arm reward seed <<< "$cfg"
  case $m in
    I) M=Qwen/Qwen2.5-1.5B-Instruct ;;
    B) M=Qwen/Qwen2.5-1.5B ;;
    *) echo "unknown model $m"; continue ;;
  esac
  name="$m-$arm-$reward-s$seed"
  mkdir -p "$OUT/$name"
  case $arm in
    base)
      evaluate "$name" "$M" ;;
    tiny)
      phase "$name/train" "$OUT/$name/train.log" $S/grpo_train.py --model "$M" --steps "$STEPS" $TINY --tiny-seed "$seed" \
        --seed "$seed" --reward "$reward" --lr "$TINY_LR" $GRPO --save-dir "$CK/$name" --save-merged --out "$OUT/$name/train" \
        && evaluate "$name" "$CK/$name/merged"
      cp "$CK/$name/tinylora.pt" "$OUT/$name/" 2>/dev/null ;;
    full)
      phase "$name/train" "$OUT/$name/train.log" $S/grpo_train.py --model "$M" --steps "$FULL_STEPS" --adapter full \
        --fp32-master --seed "$seed" --reward "$reward" --lr "$FULL_LR" $GRPO --save-dir "$CK/$name" --out "$OUT/$name/train" \
        && evaluate "$name" "$CK/$name"
      ;;
  esac
  rm -rf "${CK:?}/$name"
done
