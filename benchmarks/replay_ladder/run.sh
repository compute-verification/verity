#!/usr/bin/env bash
# The replay ladder on one node (README.md), under research run, with the C4 shard sent (--send):
#   run.sh MODEL...   for each model (ladder.LADDER): serve its saturated window (a capacity burst, then the record arm),
#                     REPLAYS compiled replays from vLLM's compile cache, FRESH more each compiled from empty caches, and
#                     ALONE of its requests served alone; then summary.json and summary.md over every model.
# VENV's vLLM must be d9105ea80, the integration's pin; the checkpoints are the ones checkpoints.json pins, under HF_HOME
# (pod_bootstrap.sh --cases ROLE,... fetches and checks them).  SPAN_S, OVER and BURST set the window (ladder.py serve).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
# FlashInfer's sampler JIT needs a newer nvcc than some images carry; greedy rows never reach it
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
REPLAYS=${REPLAYS:-3}; FRESH=${FRESH:-1}; ALONE=${ALONE:-64}
[ $# -gt 0 ] || { echo "replay-ladder: name the models (ladder.py's LADDER)"; exit 2; }
[ -f "$C4" ] || { echo "replay-ladder: no C4 shard (--send it, or C4=PATH)"; exit 2; }
v=$("$PY" -c 'import vllm; print(vllm.__version__)') || exit 2
case $v in *+gd9105ea8*) ;; *) echo "replay-ladder: vLLM $v is not d9105ea80, the integration's pin"; exit 2 ;; esac
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.max.sm,driver_version --format=csv | tee "$OUT/gpus.csv"
rc=0
for m in "$@"; do
  d=$OUT/$m; mkdir -p "$d"
  echo "== $m serve $(date -u +%FT%TZ)"
  if ! "$PY" "$here/ladder.py" serve --model "$m" --c4 "$C4" --out "$d" ${SPAN_S:+--span-s "$SPAN_S"} ${OVER:+--over "$OVER"} \
      ${BURST:+--burst "$BURST"} > "$d/serve.log" 2>&1; then
    tail -n 40 "$d/serve.log"; rc=1; continue
  fi
  grep '^\[ladder\]' "$d/serve.log"
  n=$((REPLAYS + FRESH))
  for i in $(seq 1 "$n"); do
    echo "== $m replay $i $(date -u +%FT%TZ)"
    if [ "$i" -gt "$REPLAYS" ]; then
      fresh=$(mktemp -d)
      env VLLM_DISABLE_COMPILE_CACHE=1 TORCHINDUCTOR_CACHE_DIR="$fresh/inductor" TRITON_CACHE_DIR="$fresh/triton" \
        "$PY" "$here/ladder.py" replay --model "$m" --out "$d" --index "$i" > "$d/replay-$i.log" 2>&1 || { tail -n 40 "$d/replay-$i.log"; rc=1; }
      rm -rf "$fresh"
    else
      "$PY" "$here/ladder.py" replay --model "$m" --out "$d" --index "$i" > "$d/replay-$i.log" 2>&1 || { tail -n 40 "$d/replay-$i.log"; rc=1; }
    fi
    grep '^{"mode"' "$d/replay-$i.log" | tail -n 1
  done
  if [ "$ALONE" -gt 0 ]; then
    echo "== $m alone $(date -u +%FT%TZ)"
    "$PY" "$here/ladder.py" alone --model "$m" --out "$d" --requests "$ALONE" > "$d/alone.log" 2>&1 || { tail -n 40 "$d/alone.log"; rc=1; }
    grep '^{"model"' "$d/alone.log" | tail -n 1
  fi
  gzip -f "$d"/*/server/step_trace.*.jsonl 2>/dev/null
done
"$PY" "$here/ladder.py" summary "$OUT"/*/ --out "$OUT"
exit $rc
