#!/usr/bin/env bash
# The stock control of served_verdicts.sh: its endpoint, engine and load with no protocol and no Commit, REPS times.
# Each rep is one `verity-vllm serve-openai --no-commit` lifetime of served_verdicts.sh's model (REPO at REV, --precision PRECISION),
# the bench row's scheduling (--required-manifest ROW/manifest.json, which sets only the engine's async_scheduling here),
# --max-model-len MAX_MODEL_LEN, --max-num-batched-tokens MNBT (default 512, served_verdicts.sh's T), --admit release-order and --window-requests
# SLOTS, under served_verdicts.sh's engine environment (VLLM_BATCH_INVARIANT=1, eager), then LOADS loads of
# `demos/vllm/serve_load.py --varied` with served_verdicts.sh's REQUESTS, RATE, PROMPTS, MAX_TOKENS and SEED, each under a
# release run of its own. The first load is the composed run's one load after its endpoint listens; later ones are warm.
# What the composed run has and this doesn't: the Commit (committer, taps, GPU tree, seal), H1, any SERVE_ARGS subprotocol, and
# the weights registration before the load. Each lifetime's --out is a serve dir on the node, SERVE_ROOT/runs/RUN/<lifetime>:
# serve.json, advice_trace.jsonl (the scheduler's step trace with wall times), summary.json, pauses.jsonl, load-<j>.jsonl,
# the client's record of load j, and load-<j>.timings.json, its per-request send, first-token and done times. The trace and
# the load's records hold request values, so they stay there: once the endpoint stops, `served_overhead.py metrics` reads
# them on the node and $OUT/<lifetime>/overhead.json gets its aggregate alone, which `served_overhead.py table --stock`
# reads beside the composed run's $OUT/serve/overhead.json. WARM lifetimes (default 1) run first as warm-<i>,
# so the reps find the Triton and vLLM caches as warm as the composed run's node keeps them; a cold first lifetime's first step
# took 0.40 s against 0.018 s warm (r20261027-175833-6214). With NOT_AFTER (a UTC time, `date -d` syntax) no lifetime starts
# after it, so a control queued behind other leases never runs into a later window (V2's run of record, say).
#   research run --on vy-dc-h200-2 --project verity --source TREE --cwd clone --timeout 90m -- \
#     env ROW=... bash benchmarks/one_stage/served_stock.sh
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
H=/workspace/research/vllm-composed-h200int8
ROW=${ROW:-$H/r20261026-185144-3ccb/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
VENV=${VENV:-/workspace/cp/venv312}
REPO=${REPO:-verity-int8/Llama-3.2-1B-INT8} REV=${REV:-15b0976d400d796ee589e7737d03a919526bf800} PRECISION=${PRECISION:-verity_int8}
PORT=${PORT:-18191} SLOTS=${SLOTS:-4} PROMPTS=${PROMPTS:-16:24} MAX_TOKENS=${MAX_TOKENS:-4} RATE=${RATE:-1000} SEED=${SEED:-1}
REQUESTS=${REQUESTS:-$SLOTS} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512} MNBT=${MNBT:-512} REPS=${REPS:-5} LOADS=${LOADS:-3} WARM=${WARM:-1}
export PY=$VENV/bin/python HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
ROOT=${SERVE_ROOT:-${TMPDIR:-/tmp}/served-stock}
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host

serve() {  # under the lease: lifetime $1's (rep-<i> or warm-<i>) endpoint, its LOADS loads, the endpoint stopped
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  local D=$ROOT/runs/${RESEARCH_RUN_ID:-local}/$1 srv up=0 rc=0 j release
  if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
    echo "$1: not started, past NOT_AFTER $NOT_AFTER" | tee -a "$OUT/stock.txt"; return 11
  fi
  mkdir -p "$D" "$OUT/$1"
  cd "$TREE/integrations/vllm" || return 9
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision "$PRECISION" --run-id "${RESEARCH_RUN_ID:-local}-$1" --no-commit --required-manifest "$ROW/manifest.json" \
    --admit release-order --window-requests "$SLOTS" --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MNBT" \
    > "$D/serve.log" 2>&1 &
  srv=$!
  trap 'kill -TERM "$srv" 2>/dev/null; wait "$srv"; exit 143' TERM INT
  for _ in $(seq 1 900); do
    grep -q "SERVE-OPENAI listening" "$D/serve.log" && { up=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    tail -40 "$D/serve.log"
    echo "$1: not listening: $(grep -m1 -E 'UsageError|Error|refus' "$D/serve.log" | cut -c1-400)" | tee -a "$OUT/stock.txt"
    kill -TERM "$srv" 2>/dev/null; wait "$srv"; return 10
  fi
  for j in $(seq 1 "$LOADS"); do
    release=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
    "$PY" "$TREE/demos/vllm/serve_load.py" --port "$PORT" --model "$REPO" --varied "$REQUESTS" --rate "$RATE" \
      --prompt-range "$PROMPTS" --max-tokens "$MAX_TOKENS" --release "$release" --seed "$SEED" --out "$D/load-$j.jsonl" \
      > "$D/load-$j.log" 2>&1 || rc=$?
    echo "$1 LOAD $j (exit $rc, release run $release): $(tail -1 "$D/load-$j.log" | sed -E 's/, prompts .*//' | cut -c1-300)" \
      | tee -a "$OUT/stock.txt"
  done
  trap - TERM INT
  kill -TERM "$srv" 2>/dev/null; wait "$srv"
  echo "$1: stopped (exit $?)" | tee -a "$OUT/stock.txt"
  "$PY" "$TREE/benchmarks/one_stage/served_overhead.py" metrics "$D" > "$OUT/$1/overhead.json" 2> "$D/overhead.err" \
    || { echo "$1: no aggregate: $(cut -c1-300 "$D/overhead.err")" | tee -a "$OUT/stock.txt"; rm -f "$OUT/$1/overhead.json"; rc=12; }
  return "$rc"
}
reps() {
  local i bad=0
  for i in $(seq 1 "$WARM"); do serve "warm-$i" || { bad=$?; [ "$bad" = 12 ] || return "$bad"; bad=0; }; done
  for i in $(seq 1 "$REPS"); do serve "rep-$i" || bad=$?; [ "$bad" = 11 ] && break; done
  return "$bad"
}

if [ "${1:-}" = reps ]; then reps; exit $?; fi
{ id -un; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; nproc; nvidia-smi -L; echo "row $ROW"; } | tee "$OUT/machine.txt"
for f in "$ROW/manifest.json" "$PY"; do [ -e "$f" ] || { echo "PREFLIGHT: no $f" | tee -a "$OUT/stock.txt"; exit 3; }; done
command -v gpu-lease >/dev/null || { echo "PREFLIGHT: no gpu-lease on this node" | tee -a "$OUT/stock.txt"; exit 3; }
mkdir -p "$ROOT/cache"
export OUT
GPU_LEASE_WHO=${GPU_LEASE_WHO:-bench-31-stock} gpu-lease 1 --wait --max-min "${LEASE_MIN:-60}" --mem-gb "${LEASE_MEM_GB:-64}" \
  -- bash "$TREE/benchmarks/one_stage/served_stock.sh" reps
