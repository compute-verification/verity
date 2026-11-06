#!/usr/bin/env bash
# The serve tax on one node (README.md), from the shipped tree's root under `research run`, the C4 shard sent with
# --send (or C4=PATH):
#   ARMS="compiled eager ..." RATES=light=R,medium=R,saturated=R [MODEL_ID=moonlight] [REPS=3] [SPAN_S=60]
#     [REQUESTS=N] [WINDOW_S=600] [KV_WINDOW_S=WINDOW_S] [TICK_MS=SAT,FREE | TICK_FROM=RECORD_ARM_DIR]
#     [REPLAY=LOAD [REPLAY_FROM=RECORD_ARM_DIR] [REPLAY_MODES="compiled eager"]] [MAX_MIN=240]
#     bash benchmarks/serve_tax/run.sh
#   CAPACITY=1 [REQUESTS=800] bash benchmarks/serve_tax/run.sh   # stock compiled under one burst: its capacity
#   REPLAY=LOAD REPLAY_FROM=RECORD_ARM_DIR [REPLAY_MODES=...] bash benchmarks/serve_tax/run.sh   # only replay an earlier window
# One plan (OUT/plan.jsonl, from the shard's first DOCS documents) is made before the GPUs are held; then each arm is a
# server lifetime of its own (serve_tax.py run) into OUT/<arm>, in ARMS's order, all under one gpu-lease; the tick arms
# take τ from TICK_MS, else from TICK_FROM, else from this job's own record arm, calibrated on its saturated load.
# REPLAY then replays LOAD's first rep of the record arm (by default this job's) in each of REPLAY_MODES (replay.py)
# into OUT/replay-<mode>; a mode listed again replays into OUT/replay-<mode>-<k>, its tokens compared with the first's too
# (REPLAY_MODES="compiled compiled": does the replay agree with itself?). Then OUT/table.{json,md}. MODEL_ID moonlight serves the replay-determinism benchmark's kept
# checkpoint on one GPU; glm53 the GLM-5.3 TP drivers' (benchmarks/served_commit/tp_decode.sh) on eight. VENV's vLLM
# must be d9105ea80, the integration's pin.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID=${MODEL_ID:-moonlight}
case $MODEL_ID in
  moonlight) CK=${CHECKPOINT:-/workspace/research/scratch/replay-determinism/moonlight}; GPUS=1 ;;
  glm53)
    REV=aca966e4e02791568aa6a4ced368624b3d897f42
    if [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then CK=/workspace/cp/glm53-ckpt/$REV; else CK=/workspace/fs/glm53/$REV; fi
    CK=${CHECKPOINT:-$CK}; GPUS=8 ;;
  *) echo "serve-tax: MODEL_ID is moonlight or glm53, not $MODEL_ID"; exit 2 ;;
esac
VENV=${VENV:-/workspace/cp/venv312}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"   # the venv's ninja, which FlashInfer's JIT (vLLM's default sampler) runs
# nvcc, which the kv arm builds the row-table kernels with: CUDA_HOME's, else the newest toolkit under /usr/local
command -v nvcc >/dev/null || for d in ${CUDA_HOME:+"$CUDA_HOME/bin"} $(ls -d /usr/local/cuda*/bin 2>/dev/null | sort -rV); do
  [ -x "$d/nvcc" ] && { export PATH="$PATH:$d"; break; }
done
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
REPS=${REPS:-3}; SPAN_S=${SPAN_S:-60}; MAX_MIN=${MAX_MIN:-240}; DOCS=${DOCS:-45000}
WINDOW_S=${WINDOW_S:-600}; KV_WINDOW_S=${KV_WINDOW_S:-$WINDOW_S}; PLAN=$OUT/plan.jsonl
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export HF_HOME=${HF_HOME:-/workspace/cp/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
[ -f "$C4" ] || { echo "serve-tax: no C4 shard (--send it, or C4=PATH)"; exit 2; }
[ -d "$CK" ] || { echo "serve-tax: no checkpoint at $CK"; exit 2; }
v=$("$PY" -c 'import vllm; print(vllm.__version__)') || exit 2
case $v in *+gd9105ea8*) ;; *) echo "serve-tax: vLLM $v is not d9105ea80, the integration's pin"; exit 2 ;; esac
mkdir -p "$OUT"

if [ "${1:-}" = leased ]; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.max.sm --format=csv
  common=(--model-id "$MODEL_ID" --checkpoint "$CK" --c4 "$C4")
  if [ "${CAPACITY:-0}" = 1 ]; then
    "$PY" "$here/serve_tax.py" run --arm compiled "${common[@]}" --rates burst=1000 --requests "${REQUESTS:-800}" --reps 1 \
      --warm-s 10 --out "$OUT/capacity"
    exit $?
  fi
  rc=0
  if [ -n "${TICK_MS:-}" ]; then tick=(--tick-ms "$TICK_MS"); else tick=(--tick-from "${TICK_FROM:-$OUT/record}"); fi
  count=(); [ -n "${REQUESTS:-}" ] && count=(--requests "$REQUESTS")
  for arm in ${ARMS:-}; do
    echo "== $arm $(date -u +%FT%TZ)"
    "$PY" "$here/serve_tax.py" run --arm "$arm" "${common[@]}" --rates "$RATES" --reps "$REPS" --span-s "$SPAN_S" \
      --plan "$PLAN" --docs "$DOCS" --window-s "$WINDOW_S" --kv-window-s "$KV_WINDOW_S" "${tick[@]}" "${count[@]}" \
      --out "$OUT/$arm" || rc=1
  done
  if [ -n "${REPLAY:-}" ]; then
    declare -A times=()
    for mode in ${REPLAY_MODES:-compiled eager}; do
      flag=(); [ "$mode" = eager ] && flag=(--eager)
      n=$(( ${times[$mode]:-0} + 1 )); times[$mode]=$n
      dir="replay-$mode"; [ "$n" -gt 1 ] && { dir="replay-$mode-$n"; flag+=(--against "$OUT/replay-$mode"); }
      echo "== replay $REPLAY $mode #$n $(date -u +%FT%TZ)"
      "$PY" "$here/replay.py" --arm-dir "${REPLAY_FROM:-$OUT/record}" --load "$REPLAY" --checkpoint "$CK" "${flag[@]}" \
        --out "$OUT/$dir" > "$OUT/$dir.log" 2>&1 || { tail -n 40 "$OUT/$dir.log"; rc=1; }
      tail -n 1 "$OUT/$dir.log"
    done
  fi
  exit $rc
fi

if [ "${CAPACITY:-0}" != 1 ] && [ -z "${ARMS:-}" ]; then
  [ -n "${REPLAY_FROM:-}" ] || { echo "serve-tax: ARMS: the arms to serve, in order (or only REPLAY with REPLAY_FROM)"; exit 2; }
elif [ "${CAPACITY:-0}" != 1 ]; then
  "$PY" "$here/serve_tax.py" plan --model-id "$MODEL_ID" --checkpoint "$CK" --c4 "$C4" --rates "${RATES:?RATES: NAME=REQ_PER_S,...}" \
    --span-s "$SPAN_S" ${REQUESTS:+--requests "$REQUESTS"} --docs "$DOCS" --out "$PLAN" || exit 1
fi
export PLAN DOCS WINDOW_S KV_WINDOW_S
GPU_LEASE_WHO=${GPU_LEASE_WHO:-serve-tax} gpu-lease "$GPUS" --wait --max-min "$MAX_MIN" -- bash "$here/run.sh" leased
rc=$?
gzip -f "$OUT"/*/server/step_trace.*.jsonl 2>/dev/null
dirs=()
for d in "$OUT"/*/; do [ -f "$d/meta.json" ] && dirs+=("$d"); done
[ ${#dirs[@]} -gt 0 ] && "$PY" "$here/serve_tax.py" table "${dirs[@]}" --out "$OUT"
exit $rc
