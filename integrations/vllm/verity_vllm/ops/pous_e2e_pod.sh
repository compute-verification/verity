#!/usr/bin/env bash
# `verity-vllm pous e2e` on a fresh GPU pod, as one research Attempt: the pod runtime (pod_bootstrap.sh --cpu: the venv, torch
# cu129, vLLM and the checkpoint; the run uses no native taps), then every role of the POUS end-to-end run under CODEC, with the
# flags of P3's run of record (r20260927-182518-cb49).
#
#   research run --on <pod> --project verity --source . --cwd source --campaign pous \
#     -- bash integrations/vllm/verity_vllm/ops/pous_e2e_pod.sh [CODEC=band-chain/d12/v1] [CASE=QWEN05] [AUDITS=20] [WORKERS=12]
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
TREE=$(pwd); O=$RESEARCH_RUN_DIR; WORK=${WORK:-/workspace}
CODEC=${CODEC:-band-chain/d12/v1}; CASE=${CASE:-QWEN05}; AUDITS=${AUDITS:-20}; WORKERS=${WORKERS:-12}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
(cd integrations/vllm && bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "$CASE" --out "$O/bootstrap") > "$O/bootstrap.log" 2>&1 \
  || { tail -30 "$O/bootstrap.log"; exit 3; }
PY=${VENV:-$WORK/venv312}/bin/python
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} HF_HOME=${HF_HOME:-$WORK/hf} PATH=$(dirname "$PY"):/usr/local/cuda/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$TREE/integrations/vllm:$TREE/packages/verity/src:$TREE/tools/research/src:$TREE/protocols/sampled_proofs:$TREE/protocols/pous
# torch cu129 needs a CUDA 12.9 driver; below R575 a datacenter GPU runs it through NVIDIA's forward-compatibility libraries
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -lt 575 ]; then
  if [ ! -e /usr/local/cuda-12.9/compat/libcuda.so.1 ]; then
    . /etc/os-release; CUDA_REPO=ubuntu${VERSION_ID/./}
    dpkg -s cuda-keyring >/dev/null 2>&1 || { wget -q https://developer.download.nvidia.com/compute/cuda/repos/$CUDA_REPO/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb; } || exit 3
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq --no-install-recommends cuda-compat-12-9 >/dev/null || exit 3
  fi
  export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
fi
exec verity-vllm pous e2e --out "$O/pous" --case "$CASE" --codec "$CODEC" --build-dir "$WORK/pous_build" \
  --result "$O/result.json" --run-id "$RESEARCH_RUN_ID" --verifier-cpu 3 --responder-cpu 2 --pings 5000 --audits "$AUDITS" \
  --max-tokens 6 --workers "$WORKERS"
