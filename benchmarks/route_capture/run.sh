#!/usr/bin/env bash
# The route capture (README.md), under research run on a node with eight H200s: GLM-5.3 served like production (`vllm
# serve` at vLLM's defaults: CUDA graphs, chunked prefill, prefix caching and async scheduling, with the compilation vLLM
# picks for the model; TP8, at most 256 sequences and 8,192 tokens a step) under about DURATION_S of four concurrent
# clients (`vllm bench serve`: ShareGPT greedy on a 1-to-24 request/s ramp, ShareGPT sampled at 2/s, 8k-32k-token
# prompts at 0.1/s, shared 4k-token prefixes at 1/s), with route_capture_plugin recording every step's class and the
# kernels of the first steps of each class; then the generated code of any compile (vLLM's and Inductor's Python,
# Triton's kernel metadata) beside the capture.
#   run.sh       -> $OUT: capture/ (the plugin's files), clients/ (each client's log and result), server.log, smoke.json,
#                   compiled-code.tar.gz, engine_env.txt, summary.json (routes.py summary)
# VENV's vLLM must be d9105ea80, the integration's pin. CKPT is GLM-5.3's checkpoint directory (default: the revision
# checkpoints.json pins, under HF_HOME's hub, else its local_path). SHAREGPT is ShareGPT_V3_unfiltered_cleaned_split.json
# (default HF_HOME/datasets/, fetched from the hub when missing). The FlashInfer sampler is on, vLLM's default, unless
# VLLM_USE_FLASHINFER_SAMPLER is set; if a sampled request fails with it on, the server restarts with it off and
# engine_env.txt says so. The run refuses a node whose GPUs hold memory already (FORCE=1 runs anyway).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT=$(cd "$here/../.." && pwd)
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
# DeepGEMM JIT-compiles its kernels with CUDA_HOME's nvcc: the newest toolkit under /usr/local unless CUDA_HOME names one
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -1)}
export CUDA_HOME
# vLLM takes FlashInfer's block-FP8 GEMM below 32 rows only with FlashInfer's kernels prebuilt (flashinfer-cubin, which
# production images ship) or nvcc on PATH to compile them, and its sampler compiles at warmup with that nvcc;
# engine_env.txt says which held
export PATH="$CUDA_HOME/bin:$PATH"
unset NCCL_P2P_DISABLE
# the Inductor pin the integration serves with: one reduction tiling, so the fused kernels captured are the ones replayed
export TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=${TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK:-0}
DURATION_S=${DURATION_S:-600}; PORT=${PORT:-8031}; MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}; SERVED=glm-5.3
mkdir -p "$OUT/clients"
# a client holds a socket for each request in flight, and the greedy ramp queues past a soft limit of 1024 on a saturated engine
ulimit -n "$(ulimit -Hn)"

v=$("$PY" -c 'import vllm; print(vllm.__version__)') || exit 2
case $v in *+gd9105ea8*) ;; *) echo "route-capture: vLLM $v is not d9105ea80, the integration's pin"; exit 2 ;; esac
read -r rev local_path < <(python3 - "$ROOT/adapters/vllm/manifests/checkpoints.json" <<'EOF'
import json, sys
e = next(e for e in json.load(open(sys.argv[1]))["checkpoints"] if e.get("role") == "GLM53")
print(e["revision"], e.get("local_path", ""))
EOF
)
if [ -z "${CKPT:-}" ]; then
  CKPT=$HF_HOME/hub/models--zai-org--GLM-5.3/snapshots/$rev
  [ -f "$CKPT/config.json" ] || CKPT=$local_path
fi
[ -f "$CKPT/config.json" ] || { echo "route-capture: no GLM-5.3 checkpoint at $CKPT (CKPT=DIR)"; exit 2; }
SHAREGPT=${SHAREGPT:-$HF_HOME/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}
if [ ! -s "$SHAREGPT" ]; then
  mkdir -p "$(dirname "$SHAREGPT")"
  HF_HUB_OFFLINE=0 "$PY" - "$SHAREGPT" <<'EOF' || { echo "route-capture: no ShareGPT at $SHAREGPT and the hub fetch failed (SHAREGPT=FILE)"; exit 2; }
import shutil, sys
from huggingface_hub import hf_hub_download
shutil.copy(hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                            repo_type="dataset"), sys.argv[1])
EOF
fi
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2048' | wc -l)
if [ "$busy" -gt 0 ] && [ -z "${FORCE:-}" ]; then
  echo "route-capture: $busy GPUs already hold over 2 GiB, so another job is on the node (FORCE=1 runs anyway)"; nvidia-smi; exit 2
fi

scratch=$(mktemp -d)
site=$scratch/site/verity_route_capture-0.0.dist-info; mkdir -p "$site"
printf 'Metadata-Version: 2.1\nName: verity-route-capture\nVersion: 0.0\n' > "$site/METADATA"
printf '[vllm.general_plugins]\nverity_route_capture = route_capture_plugin:register\n' > "$site/entry_points.txt"
export PYTHONPATH="$scratch/site:$here:$ROOT/adapters/vllm:$ROOT/tools/config:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# only the capture's plugin loads, and the compile starts from empty caches so the generated code kept is this run's
export VLLM_PLUGINS=verity_route_capture
export VLLM_CACHE_ROOT=$scratch/cache/vllm TORCHINDUCTOR_CACHE_DIR=$scratch/cache/inductor TRITON_CACHE_DIR=$scratch/cache/triton
export ROUTECAP_OUT=$OUT/capture

{
  echo "vllm=$v"; "$PY" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
  echo "checkpoint=$CKPT"; echo "tree=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
  echo "sharegpt_sha256=$(sha256sum "$SHAREGPT" | cut -d' ' -f1)"
  for k in VLLM_USE_FLASHINFER_SAMPLER VLLM_USE_DEEP_GEMM VLLM_ATTENTION_BACKEND VLLM_USE_V2_MODEL_RUNNER CUDA_HOME \
           TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK \
           CUDA_VISIBLE_DEVICES $(compgen -e | grep -E '^ROUTECAP_' || true); do echo "$k=${!k-unset}"; done
  "$CUDA_HOME/bin/nvcc" --version 2>/dev/null | tail -1
  "$PY" -c 'from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cubin
print(f"has_flashinfer={has_flashinfer()} flashinfer_cubin={has_flashinfer_cubin()}")' 2> /dev/null
} > "$OUT/engine_env.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.max.sm,driver_version --format=csv | tee "$OUT/gpus.csv"

SERVER=
serve() {
  echo "== serve $(date -u +%FT%TZ)" | tee -a "$OUT/server.log"
  "$PY" -m vllm.entrypoints.cli.main serve "$CKPT" --served-model-name "$SERVED" --tensor-parallel-size 8 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 256 --max-num-batched-tokens 8192 --seed 0 \
    --host 127.0.0.1 --port "$PORT" --disable-uvicorn-access-log >> "$OUT/server.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 720); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null && { echo "== ready $(date -u +%FT%TZ)"; return 0; }
    kill -0 "$SERVER" 2> /dev/null || { echo "route-capture: the server exited"; tail -n 60 "$OUT/server.log"; return 1; }
    sleep 5
  done
  echo "route-capture: the server wasn't ready in an hour"; return 1
}
stop() {
  [ -n "$SERVER" ] || return 0
  kill -INT "$SERVER" 2> /dev/null
  for _ in $(seq 1 90); do kill -0 "$SERVER" 2> /dev/null || { SERVER=; return 0; }; sleep 2; done
  kill -KILL "$SERVER" 2> /dev/null; SERVER=
}
smoke() {
  local t
  for t in 0 0.7; do
    curl -sf -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":$t,\"top_p\":0.95}" \
      >> "$OUT/smoke.json" || return 1
    echo >> "$OUT/smoke.json"
  done
}
trap 'stop; rm -rf "$scratch"' EXIT

serve || exit 1
if ! smoke; then
  stop
  [ -z "${VLLM_USE_FLASHINFER_SAMPLER+x}" ] || { echo "route-capture: the smoke requests failed"; tail -n 60 "$OUT/server.log"; exit 1; }
  echo "fallback: a sampled request failed with FlashInfer's sampler on; restarted with VLLM_USE_FLASHINFER_SAMPLER=0" \
    | tee -a "$OUT/engine_env.txt"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  mv "$OUT/capture" "$OUT/capture-sampler-on"
  serve || exit 1
  smoke || { echo "route-capture: the smoke requests failed"; tail -n 60 "$OUT/server.log"; exit 1; }
fi

bench() {
  local name=$1; shift
  "$PY" -m vllm.entrypoints.cli.main bench serve --backend openai --base-url "http://127.0.0.1:$PORT" --model "$SERVED" \
    --tokenizer "$CKPT" --trust-remote-code --seed 0 --disable-tqdm --save-result --result-dir "$OUT/clients" \
    --result-filename "$name.json" --request-id-prefix "$name-" --percentile-metrics ttft,tpot,itl,e2el "$@" \
    > "$OUT/clients/$name.log" 2>&1
}
echo "== traffic $(date -u +%FT%TZ) for about ${DURATION_S}s"
pids=()
bench sharegpt-greedy --dataset-name sharegpt --dataset-path "$SHAREGPT" --num-prompts $((DURATION_S * 72 / 10)) \
  --ramp-up-strategy linear --ramp-up-start-rps 1 --ramp-up-end-rps 24 --temperature 0 & pids+=($!)
bench sharegpt-sampled --dataset-name sharegpt --dataset-path "$SHAREGPT" --num-prompts $((DURATION_S * 2)) \
  --request-rate 2 --temperature 0.7 --top-p 0.95 & pids+=($!)
bench long-context --dataset-name random --random-input-len 20480 --random-output-len 256 \
  --random-range-ratio '{"input": 0.6, "output": 0.5}' --num-prompts $((DURATION_S / 10)) --request-rate 0.1 \
  --temperature 0.7 --top-p 0.95 & pids+=($!)
bench shared-prefix --dataset-name prefix_repetition --prefix-repetition-prefix-len 4096 --prefix-repetition-suffix-len 512 \
  --prefix-repetition-num-prefixes 16 --prefix-repetition-output-len 256 --num-prompts "$DURATION_S" --request-rate 1 \
  --temperature 0 & pids+=($!)
deadline=$(( $(date +%s) + DURATION_S + 900 ))
while :; do
  alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2> /dev/null && alive=1; done
  [ "$alive" -eq 1 ] || break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "route-capture: clients still running $((DURATION_S + 900))s in; stopping them"
    for p in "${pids[@]}"; do kill -INT "$p" 2> /dev/null; done
    break
  fi
  sleep 10
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for f in "$OUT"/clients/*.log; do echo "-- $(basename "$f")"; grep -E 'Successful requests|Failed requests|Benchmark duration|Total input tokens|Total generated' "$f"; done
echo "== stop $(date -u +%FT%TZ)"
stop

( cd "$scratch/cache" && find . -type f \( -name '*.py' -o -name '*.json' -o -name '*.txt' \) -size -16M -print0 \
    | tar --null -czf "$OUT/compiled-code.tar.gz" -T - )
gzip -f "$OUT"/capture*/steps.rank*.jsonl
"$PY" "$here/routes.py" summary "$OUT/capture" > "$OUT/summary.json" || rc=1
python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ("steps", "classes", "profiled", "kernels")}))' \
  "$OUT/summary.json" 2> /dev/null
exit $rc
