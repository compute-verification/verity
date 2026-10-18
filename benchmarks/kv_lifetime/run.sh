#!/usr/bin/env bash
# KV block lifetimes under a ShareGPT chat workload on one vLLM engine (serve.py), then their analysis (analyze.py).
# cwd = the source tree; outputs in $RESEARCH_RUN_DIR/out. ARGS go to serve.py, WARMUP (default 120 s) to analyze.py.
#
#   research run --queue --project verity --source . --cwd source --gpus 1 --preemptible --max-min 60 --mem-gb 64 \
#     --declared-output 'out/*' --env ARGS='--workload single --phases 8:1500' -- bash benchmarks/kv_lifetime/run.sh
#
# The dataset is ShareGPT_V3_unfiltered_cleaned_split.json (anon8231489123/ShareGPT_Vicuna_unfiltered), fetched into
# the node's Hugging Face cache once.
set -euxo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/jobs}; PY=$WORK/venv312/bin/python
[ -x "$PY" ] || { echo "no python at $PY"; exit 3; }
export PYTHONPATH=$PWD:$PWD/integrations/vllm HF_HOME=${HF_HOME:-/workspace/hf} CUDA_DEVICE_ORDER=PCI_BUS_ID \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 FLASHINFER_CUDA_ARCH_LIST=12.0f
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
DATA=$("$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('anon8231489123/ShareGPT_Vicuna_unfiltered', 'ShareGPT_V3_unfiltered_cleaned_split.json', repo_type='dataset'))")
sha256sum "$DATA" > "$O/sharegpt.sha256"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$O/gpus.csv" || true
# MODEL (a Hugging Face id) is served from its snapshot in the node's cache, which has no refs to resolve offline
MODEL=${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}
SNAP=$(ls -d "$HF_HOME/hub/models--${MODEL//\//--}/snapshots/"*/ | head -1)
[ -n "$SNAP" ] || { echo "no snapshot of $MODEL under $HF_HOME"; exit 3; }
HF_HUB_OFFLINE=1 "$PY" benchmarks/kv_lifetime/serve.py --out "$O" --sharegpt "$DATA" --model "$SNAP" ${ARGS:-}
"$PY" benchmarks/kv_lifetime/analyze.py "$O" --warmup "${WARMUP:-120}"
