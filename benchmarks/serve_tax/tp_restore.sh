#!/usr/bin/env bash
# A model served mid-traffic, restored at its first edge into a fresh engine and replayed to its second (tp_restore.py),
# on one node under `research run`, from the shipped tree's root, the C4 shard sent with --send (or C4=PATH):
#   [MODEL_ID=glm53] [RATE=10] [SPAN_S=240] [EDGES_AT=120,170,220] [PLAN_FILE=PLAN.jsonl] [UNSEEDED_SHARE=0.5]
#     [ARMS="mixed unseeded"] [BUDGET_MIN=90] [SAMPLING=production] [LONG_SHARE=0.03] [LONG_TOKENS=8192,32768]
#     [MAX_MODEL_LEN=40960] [KV_CACHE_DTYPE=fp8] [FLASHINFER=0] [LDCONFIG=1] (--send fi_smoke.py: run it first) [SAVER=boundary:save_edge]
#     [LOADER=boundary:load_edge] [PERMUTE=1] [KEEP=1] [MAX_MIN=150] [DUMP=1] [NUM_GPU_BLOCKS=N] [RESTORE=0] [THEN=SCRIPT]
#     bash benchmarks/serve_tax/tp_restore.sh
# PLAN_FILE, a ready plan (the replay ladder's production plan: ShareGPT, multi-turn and long prompts, each request greedy
# or sampled), is served as it is in each of ARMS: ``mixed``, each request with its own sampling and UNSEEDED_SHARE of the
# sampled ones unseeded; ``unseeded``, every request with the generation config's sampling and no seed, which vLLM samples
# with FlashInfer's sampler. Without PLAN_FILE the one arm is SAMPLING, its plan (OUT/plan.jsonl) made before the GPUs are
# held, LONG_SHARE of its requests then given prompts of 8k-32k tokens (tp_restore.lengthen). In each arm one server
# lifetime serves the plan with the edges into OUT/ARM/serve, and one in-process engine restores at the first edge and
# replays to the second into OUT/ARM/restore (its verdict OUT/ARM/restore/restore.json). An arm after the first starts only
# if the minutes used so far and the last arm's fit in BUDGET_MIN (OUT/arms.txt says which ran). PERMUTE=1 restores again
# with the GPUs in reverse order into OUT/ARM/restore-permuted; RESTORE=0 restores nothing. The first DUMP edges' blocks
# (tens of GB at TP8; NUM_GPU_BLOCKS caps the pool) go to a scratch directory outside the run's, removed at the end
# unless KEEP=1. THEN, a script, runs under the same lease after each arm as `bash SCRIPT OUT/ARM/serve SCRATCH/ARM ARM`
# (PY and HERE, this directory, set), logging to OUT/ARM/then.log. The GPUs are held under gpu-lease where the node has
# it, or under `research run --gpus N` by the run's own lease. VENV's vLLM must be d9105ea80, the integration's pin.
# MODEL_ID=moonlight (8,192 tokens of context: LONG_TOKENS=4096,7680 MAX_MODEL_LEN=8192) is a smoke of the same path
# on one GPU, TP=2 on two ranks.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID=${MODEL_ID:-glm53}
case $MODEL_ID in
  moonlight) CK=${CHECKPOINT:-/workspace/research/scratch/replay-determinism/moonlight}; GPUS=${TP:-1} ;;
  glm53)
    REV=aca966e4e02791568aa6a4ced368624b3d897f42
    if [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then CK=/workspace/cp/glm53-ckpt/$REV; else CK=/workspace/fs/glm53/$REV; fi
    CK=${CHECKPOINT:-$CK}; GPUS=8 ;;
  *) echo "tp-restore: MODEL_ID is moonlight or glm53, not $MODEL_ID"; exit 2 ;;
esac
[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312   # a dc node's, else pod_bootstrap.sh's
VENV=${VENV:-$venv}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
RATE=${RATE:-10}; SPAN_S=${SPAN_S:-240}; EDGES_AT=${EDGES_AT:-120,170,220}; SAMPLING=${SAMPLING:-production}
PLAN_FILE=${PLAN_FILE:-}; UNSEEDED_SHARE=${UNSEEDED_SHARE:-0.5}; BUDGET_MIN=${BUDGET_MIN:-90}
[ -z "$PLAN_FILE" ] || [ -f "$PLAN_FILE" ] || { echo "tp-restore: no plan at PLAN_FILE=$PLAN_FILE"; exit 2; }
if [ -n "$PLAN_FILE" ]; then ARMS=${ARMS:-mixed unseeded}; else ARMS=$SAMPLING; fi
LONG_SHARE=${LONG_SHARE:-0.03}; LONG_TOKENS=${LONG_TOKENS:-8192,32768}; MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
# the scheduler's, block pool's and each rank's runner state at an edge, and its restore (boundary.py); LOADER= (empty)
# restores only the blocks and generators and replays nothing
SAVER=${SAVER-boundary:save_edge}; LOADER=${LOADER-boundary:load_edge}; DUMP=${DUMP:-1}; RESTORE=${RESTORE:-1}; THEN=${THEN:-}
[ -z "$THEN" ] || [ -f "$THEN" ] || { echo "tp-restore: no script at THEN=$THEN"; exit 2; }
DOCS=${DOCS:-45000}; MAX_MIN=${MAX_MIN:-150}; PLAN=$OUT/plan.jsonl
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
[ -d /workspace/cp/hf ] && hf=/workspace/cp/hf || hf=/workspace/hf
export HF_HOME=${HF_HOME:-$hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
# DeepGEMM and FlashInfer JIT-compile with CUDA_HOME's nvcc: the newest toolkit under /usr/local unless CUDA_HOME names one.
# vLLM uses FlashInfer only with flashinfer-cubin installed or nvcc on PATH; FLASHINFER=1 (the default, as a production
# install with a CUDA toolkit has it) puts the nvcc on PATH: FlashInfer's sampler, autotune and fused all-reduce + RMSNorm.
# FLASHINFER=0 leaves it off, and its sampler too.
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -1)}
export CUDA_HOME
# FlashInfer's JIT finds CUDA_HOME's headers and libraries itself: an inherited CPATH or LIBRARY_PATH could mix toolkits
unset CPATH LIBRARY_PATH
if [ "${FLASHINFER:-1}" = 1 ]; then export PATH="$CUDA_HOME/bin:$PATH"
else export VLLM_USE_FLASHINFER_SAMPLER=0; fi
[ -f "$C4" ] || { echo "tp-restore: no C4 shard (--send it, or C4=PATH)"; exit 2; }
[ -d "$CK" ] || { echo "tp-restore: no checkpoint at $CK"; exit 2; }
v=$("$PY" -c 'import vllm; print(vllm.__version__)') || exit 2
case $v in *+gd9105ea8*) ;; *) echo "tp-restore: vLLM $v is not d9105ea80, the integration's pin"; exit 2 ;; esac
mkdir -p "$OUT"
if [ "${1:-}" != leased ]; then
  # LDCONFIG=1: refresh the linker cache first, so libnvrtc and the rest resolve to the CUDA_HOME a node's dev packages
  # were installed for, the same for every engine of this job
  [ "${LDCONFIG:-0}" = 1 ] && ldconfig
  for k in FLASHINFER VLLM_USE_FLASHINFER_SAMPLER VLLM_USE_DEEP_GEMM CUDA_HOME CUDA_VISIBLE_DEVICES; do echo "$k=${!k-unset}"; done \
    > "$OUT/engine_env.txt"
  { command -v nvcc && nvcc --version | tail -1
    echo "has_flashinfer=$("$PY" -c 'from vllm.utils.flashinfer import has_flashinfer; print(has_flashinfer())')"
    ldconfig -p | grep -m2 'libnvrtc.so.12 '; } >> "$OUT/engine_env.txt" 2>&1
fi
[ "${1:-}" = leased ] || "$PY" "$here/tp_restore.py" preflight --model-id "$MODEL_ID" --checkpoint "$CK" \
  ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} ${TP:+--tp "$TP"} ${KV_CACHE_DTYPE:+--kv-cache-dtype "$KV_CACHE_DTYPE"} || exit 2

if [ "${1:-}" = leased ]; then
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.total --format=csv
  # vllm's fi_smoke.py, when sent: FlashInfer's small-batch FP8 GEMMs compiled (into ~/.tensorrt_llm/cache) before the
  # first engine, so no compile lands inside the served traffic; the cache's digest is the build's pin
  smoke=$(ls "$D"/inputs/fi_smoke.py 2>/dev/null)
  if [ -n "$smoke" ] && [ "${FLASHINFER:-1}" = 1 ]; then
    echo "== fi_smoke $(date -u +%FT%TZ)"
    GPUS=$GPUS "$PY" "$smoke" "$OUT/fi_smoke.json" > "$OUT/fi_smoke.log" 2>&1 || { tail -n 20 "$OUT/fi_smoke.log"; exit 1; }
  fi
  c=$HOME/.tensorrt_llm/cache
  [ -d "$c" ] && echo "tensorrt_llm_cache=$(cd "$c" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64) files=$(find "$c" -type f | wc -l)" >> "$OUT/engine_env.txt"
  rc=0; t0=$(date +%s); last=0
  for arm in $ARMS; do
    t=$(date +%s)
    if [ "$last" -gt 0 ] && [ $((t - t0 + last)) -gt $((BUDGET_MIN * 60)) ]; then
      echo "== $arm skipped $(date -u +%FT%TZ): $(((t - t0) / 60)) min used, the last arm took $((last / 60)), BUDGET_MIN=$BUDGET_MIN" \
        | tee -a "$OUT/arms.txt"
      continue
    fi
    case $arm in
      mixed) plan=(--plan-file "$PLAN_FILE" --sampling own --unseeded-share "$UNSEEDED_SHARE") ;;
      unseeded) plan=(--plan-file "$PLAN_FILE" --sampling production) ;;
      *) plan=(--plan "$PLAN" --sampling "$arm" --long-share "$LONG_SHARE" --long-tokens "$LONG_TOKENS") ;;
    esac
    echo "== $arm serve $(date -u +%FT%TZ)"
    "$PY" "$here/tp_restore.py" serve --model-id "$MODEL_ID" --checkpoint "$CK" --c4 "$C4" "${plan[@]}" --docs "$DOCS" \
      --rate "$RATE" --span-s "$SPAN_S" --edges-at "$EDGES_AT" \
      ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} ${TP:+--tp "$TP"} ${KV_CACHE_DTYPE:+--kv-cache-dtype "$KV_CACHE_DTYPE"} \
      ${SAVER:+--saver "$SAVER"} --dump "$DUMP" \
      ${NUM_GPU_BLOCKS:+--num-gpu-blocks "$NUM_GPU_BLOCKS"} --out "$OUT/$arm/serve" --scratch "$SCRATCH/$arm/serve" \
      || { echo "== $arm serve failed $(date -u +%FT%TZ)" | tee -a "$OUT/arms.txt"; exit 1; }
    restores=(); [ "$RESTORE" = 1 ] && restores=(restore ${PERMUTE:+restore-permuted})
    for r in "${restores[@]}"; do
      echo "== $arm $r $(date -u +%FT%TZ)"
      flag=(); env=()
      if [ "$r" = restore-permuted ]; then
        flag=(--rank-map-free)
        env=(CUDA_VISIBLE_DEVICES="$(echo "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}" | tr , '\n' | tac | paste -sd,)")
      fi
      env "${env[@]}" "$PY" "$here/tp_restore.py" restore --served "$OUT/$arm/serve" --scratch "$SCRATCH/$arm/$r" \
        --out "$OUT/$arm/$r" ${LOADER:+--loader "$LOADER"} "${flag[@]}" > "$OUT/$arm/$r.log" 2>&1 \
        || { tail -n 40 "$OUT/$arm/$r.log"; rc=1; }
      tail -n 1 "$OUT/$arm/$r.log"
    done
    if [ -n "$THEN" ]; then
      echo "== $arm then $(date -u +%FT%TZ)"
      PY=$PY HERE=$here bash "$THEN" "$OUT/$arm/serve" "$SCRATCH/$arm" "$arm" > "$OUT/$arm/then.log" 2>&1 \
        || { tail -n 40 "$OUT/$arm/then.log"; rc=1; }
      tail -n 3 "$OUT/$arm/then.log"
    fi
    du -sh "$SCRATCH/$arm" 2>/dev/null
    [ "${KEEP:-0}" = 1 ] || rm -rf "$SCRATCH/$arm"
    last=$(($(date +%s) - t))
    echo "== $arm done $(date -u +%FT%TZ) in $((last / 60)) min" | tee -a "$OUT/arms.txt"
  done
  exit $rc
fi

[ -n "$PLAN_FILE" ] || "$PY" "$here/serve_tax.py" plan --model-id "$MODEL_ID" --checkpoint "$CK" --c4 "$C4" \
  --rates "load=$RATE" --span-s "$SPAN_S" --docs "$DOCS" --out "$PLAN" || exit 1
root=/workspace/research/scratch; [ -d "$root" ] || root=${TMPDIR:-/tmp}
SCRATCH=${SCRATCH:-$(mktemp -d "$root/tp-restore.XXXXXX")}
export SCRATCH
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  # already under a lease (`research run --gpus N`), which must hold this job's GPUs: leasing again would wait on itself
  n=$(echo "$GPU_LEASE_UUID" | tr , '\n' | grep -c .)
  if [ "$n" -lt "$GPUS" ]; then echo "tp-restore: the run's lease holds $n GPU(s), not $GPUS"; rc=2
  else bash "$here/tp_restore.sh" leased; rc=$?; fi
elif command -v gpu-lease > /dev/null; then
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-tp-restore} gpu-lease "$GPUS" --wait --max-min "$MAX_MIN" -- bash "$here/tp_restore.sh" leased
  rc=$?
else
  # a node without gpu-lease queues its jobs by other means: refuse to start on GPUs another job still holds
  for _ in $(seq 31); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt "${FREE_MIB:-2048}" ] && break
    sleep 20
  done
  if [ "${used:-0}" -ge "${FREE_MIB:-2048}" ]; then
    echo "tp-restore: a GPU still holds $used MiB after 10 min; not starting"; rc=75
  else
    bash "$here/tp_restore.sh" leased; rc=$?
  fi
fi
gzip -f "$OUT"/*/serve/server/step_trace.*.jsonl "$OUT"/*/serve/edges/outputs.jsonl 2>/dev/null
du -sh "$SCRATCH" 2>/dev/null
[ "${KEEP:-0}" = 1 ] || rm -rf "$SCRATCH"
echo "tp-restore: done rc=$rc $(date -u +%FT%TZ); GPU memory used (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,)"
exit $rc
