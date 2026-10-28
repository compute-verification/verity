#!/usr/bin/env bash
# One chunk of per-token vLLM timing traces on one leased GPU (record.py's docstring), from the shipped tree's root:
#   research run --queue --source <tree> --cwd source --project verity --gpus 1 --max-min 60 --cpus 8 --mem-gb 48 \
#     --declared-output 'out/*' [--env MODEL=...] [--env MIX=b16=3,b4=1,b64=10] [--env MODE=plaintext|commit|bi-eager] -- \
#     bash benchmarks/network_traces/run.sh [--server-arg=--enforce-eager ...]
# MINUTES is the whole job, startup included, and must stay under the lease's --max-min. The node's vLLM venv (VENV) and
# Hugging Face cache (HF_HOME, offline) are used as they are.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}
mkdir -p "$OUT"
VENV=${VENV:-/workspace/jobs/venv312}
export PATH=$VENV/bin:$PATH HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.0f}
: "${CUDA_VISIBLE_DEVICES:?run this under a GPU lease}"
case "${MODE:-plaintext}" in
  plaintext) ;;
  *) export PYTHONPATH=$PWD/integrations/vllm:$PWD:$PWD/tools/research/src${PYTHONPATH:+:$PYTHONPATH} ;;
esac
nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES%%,*}" > "$OUT/nvidia-smi.txt" 2>&1 || true
exec "$VENV/bin/python" benchmarks/network_traces/record.py run --out "$OUT" --model "${MODEL:-unsloth/Llama-3.2-1B}" \
  --mix "${MIX:-b16=3,b4=1,b64=10}" --mode "${MODE:-plaintext}" --minutes "${MINUTES:-55}" "$@"
