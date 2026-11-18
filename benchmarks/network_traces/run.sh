#!/usr/bin/env bash
# One chunk of per-token vLLM timing traces on one GPU (record.py's docstring), from a clone of the tree, which prints
# and records its one number (record.py's `METRIC`). Its outputs go to OUT (default $RESEARCH_RUN_DIR/out, else ./out):
#   [MODEL=...] [MIX=b16=3,b4=1,b64=10] [MODE=plaintext|commit|bi-eager] [MINUTES=55] verity bench network_traces \
#     [--server-arg=--enforce-eager ...]
#   research run --on NODE --project verity --source . --cwd source --as network-accounting --timeout 90m -- \
#     bash benchmarks/network_traces/run.sh --on CARD --cores CORES
# MINUTES is the whole recording, startup included. The GPU is the caller's gpu-lease's, else one this takes with
# GPU_LEASE (default gpu-lease) where the node has it, on the card CARD pinned to the CPUs CORES, else
# CUDA_VISIBLE_DEVICES's. FlashInfer's build is the card's (benchmarks/harness/flashinfer_arch.sh). The pinned-vLLM venv and the
# Hugging Face cache (offline) are benchmarks/harness/env.sh's, under WORK; where it finds no venv, pod_bootstrap.sh
# installs it and CASES's checkpoints (default LLAMA32_1B, the default model's role) there first, before a GPU is held.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
if ! (. benchmarks/harness/env.sh) > /dev/null 2>&1 && [ -z "${VERITY_BENCH_VENV:-}" ]; then
  echo "bootstrap: no pinned-vLLM venv, so pod_bootstrap.sh installs one under ${WORK:-/workspace}" \
    "(its log is $OUT/bootstrap.log)"
  ROW_TAPS=0 VENV=${WORK:-/workspace}/venv312 bash adapters/vllm/verity_vllm/ops/pod_bootstrap.sh \
    --cases "${CASES:-LLAMA32_1B}" --gpu --out "$OUT/bootstrap" > "$OUT/bootstrap.log" 2>&1 \
    || { echo "run.sh: the bootstrap failed, so this run has no number (its log is $OUT/bootstrap.log)" >&2; exit 1; }
fi
CARD=; CORES=; ARGS=()
while [ $# -gt 0 ]; do
  case $1 in --on) CARD=${2:?--on CARD}; shift 2 ;; --cores) CORES=${2:?--cores CORES}; shift 2 ;; *) ARGS+=("$1"); shift ;; esac
done
MINUTES=${MINUTES:-55}
read -ra LEASE <<< "${GPU_LEASE:-gpu-lease}"
if [ -z "${GPU_LEASE_UUID:-}" ] && command -v "${LEASE[0]}" > /dev/null; then
  exec env GPU_LEASE_WHO="${GPU_LEASE_WHO:-network-traces}" "${LEASE[@]}" 1 --wait ${CARD:+--on "$CARD"} \
    ${CORES:+--cores "$CORES"} --max-min "$(( ${MINUTES%.*} + 5 ))" -- bash benchmarks/network_traces/run.sh "${ARGS[@]}"
fi
: "${CUDA_VISIBLE_DEVICES:?no GPU: run this under gpu-lease, or name the card in CUDA_VISIBLE_DEVICES}"
# shellcheck source=/dev/null
. benchmarks/harness/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
. benchmarks/harness/flashinfer_arch.sh
nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES%%,*}" > "$OUT/nvidia-smi.txt" 2>&1 || true
exec "$VENV/bin/python" benchmarks/network_traces/record.py run --out "$OUT" --model "${MODEL:-unsloth/Llama-3.2-1B}" \
  --mix "${MIX:-b16=3,b4=1,b64=10}" --mode "${MODE:-plaintext}" --minutes "$MINUTES" "${ARGS[@]}"
