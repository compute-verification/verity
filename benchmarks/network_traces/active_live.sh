#!/usr/bin/env bash
# The active certifier in front of a live vLLM on one leased GPU (active_replay.py's "A live server"), from the shipped
# tree's root, as a job of at most 45 minutes whose inputs are calibration.json and NAME.trace.json.gz and whose outputs
# are out/*:
#   [MODEL=...] [VLLM_CPUS=96-103] [MINUTES=40] bash benchmarks/network_traces/active_live.sh [--config NAME ...]
# MINUTES bounds the job, the server's startup included: no configuration starts that would end after it. The node's
# vLLM venv (VENV) and Hugging Face cache (HF_HOME, offline) are used as they are, as run.sh uses them.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}
IN=${IN:-$RESEARCH_RUN_DIR/inputs}
mkdir -p "$OUT"
VENV=${VENV:-/workspace/jobs/venv312}
export PATH=$VENV/bin:$PATH HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.0f}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
: "${CUDA_VISIBLE_DEVICES:?run this under a GPU lease}"
nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES%%,*}" > "$OUT/nvidia-smi.txt" 2>&1 || true
exec "$VENV/bin/python" benchmarks/network_traces/active_replay.py --vllm "${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}" \
  ${VLLM_CPUS:+--vllm-cpus "$VLLM_CPUS"} --calibration "$IN/calibration.json" --out "$OUT" \
  --max-minutes "${MINUTES:-40}" "$@" "$IN"/*.trace.json.gz
