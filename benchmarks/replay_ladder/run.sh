#!/usr/bin/env bash
# The replay ladder on one node (README.md), under research run, with the C4 shard sent (--send):
#   run.sh MODEL...   for each model (ladder.LADDER) and each arm of ARMS: serve its saturated window (a capacity burst,
#                     then the record arm), replays of it compiled, and some of its requests served alone; then
#                     summary.json and summary.md over every model and arm.
# The arms (LADDER_ARM, ladder.pin): `pinned` serves and replays with each of Inductor's on-device choices pinned to its
# first candidate (inductor_pin.py): REPLAYS replays from vLLM's compile cache, FRESH more each compiled from
# empty caches, and ALONE requests alone. `unpinned` serves and replays without it, and `rblock` with only the
# halved-R0_BLOCK candidate dropped (TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0): UNPINNED_REPLAYS replays each, which say
# whether the model needs the pin. Each arm compiles into caches of its own: a compile cache keeps the choices made where
# it was written, so a graph compiled in one arm and loaded in another would keep the first arm's.
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
# DeepGEMM, vLLM's FP8 block GEMM, JIT-compiles its kernels with CUDA_HOME's nvcc, which CUDA 12.4's can't do: the newest
# toolkit under /usr/local unless CUDA_HOME names one. DEEP_GEMM=0 serves FP8 without DeepGEMM.
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -1)}
export CUDA_HOME
unset NCCL_P2P_DISABLE
if [ -n "${DEEP_GEMM:-}" ]; then export VLLM_USE_DEEP_GEMM=$DEEP_GEMM; else unset VLLM_USE_DEEP_GEMM; fi
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
ARMS=${ARMS:-pinned unpinned}
# the arms whose engines also run at TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 (rblock always does; never unpinned)
RBLOCK0_ARMS=${RBLOCK0_ARMS:-rblock}
REPLAYS=${REPLAYS:-3}; FRESH=${FRESH:-1}; ALONE=${ALONE:-64}; UNPINNED_REPLAYS=${UNPINNED_REPLAYS:-3}
[ $# -gt 0 ] || { echo "replay-ladder: name the models (ladder.py's LADDER)"; exit 2; }
[ -f "$C4" ] || { echo "replay-ladder: no C4 shard (--send it, or C4=PATH)"; exit 2; }
for arm in $ARMS; do case $arm in pinned|rblock|unpinned) ;; *) echo "replay-ladder: ARMS takes pinned, rblock and unpinned, not $arm"; exit 2 ;; esac; done
for arm in $RBLOCK0_ARMS; do case $arm in pinned|rblock) ;; *) echo "replay-ladder: RBLOCK0_ARMS takes pinned and rblock, not $arm"; exit 2 ;; esac; done
v=$("$PY" -c 'import vllm; print(vllm.__version__)') || exit 2
case $v in *+gd9105ea8*) ;; *) echo "replay-ladder: vLLM $v is not d9105ea80, the integration's pin"; exit 2 ;; esac
mkdir -p "$OUT"
for k in VLLM_USE_DEEP_GEMM VLLM_USE_FLASHINFER_SAMPLER CUDA_HOME CUDA_VISIBLE_DEVICES; do echo "$k=${!k-unset}"; done > "$OUT/engine_env.txt"
"$CUDA_HOME/bin/nvcc" --version 2>/dev/null | tail -1 >> "$OUT/engine_env.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.max.sm,driver_version --format=csv | tee "$OUT/gpus.csv"
rc=0
# the FlashInfer autotune caches a process left (tactics its warmup chose by timing), by digest, after each process
autotune() { find "$cache" "$HOME/.cache/flashinfer" -type f -iname '*autotun*' -exec sha256sum {} + 2>/dev/null | sed "s|^|$1 |" >> "$d/autotune.sha256"; }
for m in "$@"; do
  for arm in $ARMS; do
    d=$OUT/$arm/$m; mkdir -p "$d"
    cache=$(mktemp -d)
    export VLLM_CACHE_ROOT=$cache/vllm TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
    unset TORCHINDUCTOR_DETERMINISTIC TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK
    export LADDER_ARM=$arm
    replays=$UNPINNED_REPLAYS; fresh=0; alone=0
    [ "$arm" = pinned ] && { replays=$REPLAYS; fresh=$FRESH; alone=$ALONE; }
    case " $RBLOCK0_ARMS rblock " in *" $arm "*) export TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 ;; esac
    echo "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=${TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK-unset}" > "$d/arm_env.txt"
    echo "== $m $arm serve $(date -u +%FT%TZ) $(cat "$d/arm_env.txt")"
    if ! "$PY" "$here/ladder.py" serve --model "$m" --c4 "$C4" --out "$d" ${SPAN_S:+--span-s "$SPAN_S"} ${OVER:+--over "$OVER"} \
        ${BURST:+--burst "$BURST"} ${PLAN:+--plan "$PLAN"} ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} > "$d/serve.log" 2>&1; then
      tail -n 40 "$d/serve.log"; rc=1; rm -rf "$cache"; continue
    fi
    grep '^\[ladder\]' "$d/serve.log"
    autotune serve
    for i in $(seq 1 $((replays + fresh))); do
      echo "== $m $arm replay $i $(date -u +%FT%TZ)"
      if [ "$i" -gt "$replays" ]; then
        f=$(mktemp -d)
        env VLLM_DISABLE_COMPILE_CACHE=1 TORCHINDUCTOR_CACHE_DIR="$f/inductor" TRITON_CACHE_DIR="$f/triton" \
          "$PY" "$here/ladder.py" replay --model "$m" --out "$d" --index "$i" > "$d/replay-$i.log" 2>&1 || { tail -n 40 "$d/replay-$i.log"; rc=1; }
        rm -rf "$f"
      else
        "$PY" "$here/ladder.py" replay --model "$m" --out "$d" --index "$i" > "$d/replay-$i.log" 2>&1 || { tail -n 40 "$d/replay-$i.log"; rc=1; }
      fi
      grep '^{"mode"' "$d/replay-$i.log" | tail -n 1
      autotune "replay-$i"
    done
    if [ "$alone" -gt 0 ]; then
      echo "== $m $arm alone $(date -u +%FT%TZ)"
      "$PY" "$here/ladder.py" alone --model "$m" --out "$d" --requests "$alone" > "$d/alone.log" 2>&1 || { tail -n 40 "$d/alone.log"; rc=1; }
      grep '^{"model"' "$d/alone.log" | tail -n 1
    fi
    gzip -f "$d"/*/server/step_trace.*.jsonl 2>/dev/null
    rm -rf "$cache"
  done
done
unset LADDER_ARM
"$PY" "$here/ladder.py" summary "$OUT"/*/*/ --out "$OUT"
exit $rc
