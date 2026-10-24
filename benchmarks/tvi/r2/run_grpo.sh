#!/usr/bin/env bash
# R2 (elicitation controls), one leased GPU walking a node-local queue of GRPO items on GSM8K.
# ITEMS="M:method:reward:seed:steps[:task] ..." (task gsm8k, the default, or math) with M = I (Qwen2.5-1.5B-Instruct) or B (Qwen2.5-1.5B); every GPU job on a node
# claims the first unclaimed item in $QDIR (mkdir is atomic) and skips one that would end past $DEADLINE (epoch seconds).
# Sent inputs: grpo.py and the {gsm8k,math}_{train,test}.jsonl A2 uses. Results and logs go to $OUT; a TinyLoRA item's trained
# state (state.pt, meta.json: the 2,492 fp32 coefficients) is copied to $OUT/state/; checkpoints are scratch.
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
mkdir -p /workspace/jobs/tvi
S=$(mktemp -d -p /workspace/jobs/tvi .r2-run.XXXXXX)
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$IN"/grpo.py "$S"/
mkdir -p "$S/data" "$S/ckpt"
cp "$IN"/*.jsonl "$S/data/"
cd "$S"

V=/workspace/jobs/tvi/venv-R2
(
  flock 9
  if [ ! -f "$V/READY" ]; then
    rm -rf "$V"
    uv venv -q -p 3.12 "$V" && uv pip install -q --python "$V/bin/python" vllm math-verify ninja && date -u +%FT%TZ > "$V/READY"
  fi
) 9>/workspace/jobs/tvi/.venv-R2.lock
research keep "$V" --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program (R2)' >/dev/null 2>&1 || true
PY="$V/bin/python"
export PATH="$V/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else
  mkdir -p /workspace/jobs/tvi/hf-R2 && export HF_HOME=/workspace/jobs/tvi/hf-R2
  research keep /workspace/jobs/tvi/hf-R2 --owner @training --expires 2026-10-27 --priority low --why 'training-vs-inference program (R2 HF cache)' >/dev/null 2>&1 || true
fi
SVD=/workspace/jobs/tvi/cache/R2
mkdir -p "$SVD"
research keep "$SVD" --owner @training --expires 2026-10-27 --priority low --why 'training-vs-inference program (R2 SVD cache)' >/dev/null 2>&1 || true
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,name,memory.used,memory.total --format=csv
"$PY" -c 'import torch, transformers, vllm, math_verify; print(torch.__version__, transformers.__version__, vllm.__version__, torch.cuda.get_device_name(0))' | tee -a "$OUT/versions.txt"
set +e

run_item() {  # M:method:reward:seed:steps[:task]
  IFS=: read -r m method reward seed steps task <<< "$1"
  task=${task:-gsm8k}
  case $m in
    I) model=Qwen/Qwen2.5-1.5B-Instruct; variant="" ;;
    B) model=Qwen/Qwen2.5-1.5B; variant=B ;;
  esac
  local log="$OUT/log-$task-$m-$method-$reward-seed$seed.txt"
  "$PY" grpo.py --task "$task" --method "$method" --reward "$reward" --seed "$seed" --steps "$steps" --model "$model" \
    ${variant:+--variant $variant} --data "$S/data" --out "$OUT" --ckpt-root "$S/ckpt" --svd-cache "$SVD" > "$log" 2>&1
  local rc=$?
  echo "$(date -u +%H:%MZ) $1 rc=$rc"
  [ $rc -eq 0 ] || { echo "rc=$rc ${RESEARCH_RUN_ID:-}" > "$QDIR/$name/failed"; tail -20 "$log"; }
  grep -h '"event": "\(eval\|done\)"' "$log" | tail -3
  for st in "$S"/ckpt/*/step*/state.pt; do
    [ -e "$st" ] || continue
    d="$OUT/state/$(basename "$(dirname "$(dirname "$st")")")/$(basename "$(dirname "$st")")"
    mkdir -p "$d" && cp "$st" "$(dirname "$st")/meta.json" "$d/"
  done
  rm -rf "${S:?}"/ckpt/*
  return $rc
}

: "${QDIR:?set QDIR=/workspace/jobs/tvi/r2-queue/NAME}"
mkdir -p "$QDIR"
fails=0
while :; do
  picked=""
  for item in $ITEMS; do
    name="${item//:/-}"
    [ -e "$QDIR/$name" ] && continue
    IFS=: read -r _ method _ _ _ <<< "$item"
    est=$([ "$method" = full ] && echo 35 || echo 25)
    if [ -n "${DEADLINE:-}" ] && [ $(( $(date +%s) + est * 60 )) -gt "$DEADLINE" ]; then continue; fi
    if mkdir "$QDIR/$name" 2>/dev/null; then
      echo "${RESEARCH_RUN_ID:-} $(date -u +%H:%MZ)" > "$QDIR/$name/claimed"
      picked="$item"
      break
    fi
  done
  [ -z "$picked" ] && break
  if run_item "$picked"; then fails=0; else fails=$((fails + 1)); fi
  touch "$QDIR/$name/finished"
  # Two failures in a row on this GPU: stop rather than drain the queue the other GPUs share.
  [ $fails -lt 2 ] || { echo "two failed items in a row; stopping"; exit 4; }
done
echo "queue empty $(date -u +%FT%TZ)"
