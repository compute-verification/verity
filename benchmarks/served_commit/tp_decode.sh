#!/usr/bin/env bash
# One arm of tp_decode.py (stock against the arm in one TP engine, eager), from the source tree's root on an 8-GPU H200 node:
#   ARM=rows RESEARCH_RUN_DIR=DIR bash benchmarks/served_commit/tp_decode.sh
# ARM: rows | rows-4k | rows-layer | rows-layer-4k.  RETAIN (default none): the committer's value retention, VERITY_RETAIN
# (none | host | device), which summary.json records as the committer took it.  BATCHES (default 1,8,32), PAIRS (default 3), ROW
# (the workload, default GLM-5.3's TP8 eager B1, whose request every batch rotates), CASE (default GLM53), TP (default 8), VENV (default
# /workspace/cp/venv312).  The engine: gpu_memory_utilization GPU_UTIL (0.92), max_model_len MAX_MODEL_LEN (288),
# max_num_batched_tokens MAX_BATCHED (2048, the workload's).
# Out: $RESEARCH_RUN_DIR/out/$ARM (runs.jsonl, summary.json, committer_make.json, versions.json, b<B>/ the warm-ups).
set -euo pipefail
TREE=$PWD
ARM=${ARM:?ARM: rows | rows-4k | rows-layer | rows-layer-4k}
ROW=${ROW:-glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager}
REV=aca966e4e02791568aa6a4ced368624b3d897f42
FS_CK=/workspace/fs/glm53/$REV
LOCAL_CK=/workspace/cp/glm53-ckpt/$REV
if [ -e "$LOCAL_CK/.verified" ] && ! mountpoint -q "$FS_CK"; then sudo -n mount --bind "$LOCAL_CK" "$FS_CK"; fi
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export HF_HOME=/workspace/cp/hf HF_HUB_OFFLINE=1 NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
export VERITY_WINDOW_MB=256 VERITY_WINDOW_SLOTS=8 VERITY_RETAIN=${RETAIN:-none} VERITY_WEIGHTS_HASH=device
TMPDIR=$(mktemp -d /tmp/tp-decode.XXXXXX)
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT
# shellcheck disable=SC1091
source "${VENV:-/workspace/cp/venv312}/bin/activate"
export PYTHONPATH=$TREE:$TREE/benchmarks:$TREE/integrations/vllm:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/benchmarks/numerical/python:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
cd "$TREE/integrations/vllm"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
python -c "from verity_vllm.commit.committer import native_collect as N; N.ext()"
python -u -m served_commit.tp_decode --workload "workloads/$ROW.json" --case "${CASE:-GLM53}" --tp "${TP:-8}" --arm "$ARM" \
  --batches "${BATCHES:-1,8,32}" --pairs "${PAIRS:-3}" --engine-arg "gpu_memory_utilization=${GPU_UTIL:-0.92}" \
  --engine-arg "max_model_len=${MAX_MODEL_LEN:-288}" --engine-arg "max_num_batched_tokens=${MAX_BATCHED:-2048}" \
  --engine-arg disable_custom_all_reduce=false --out "$RESEARCH_RUN_DIR/out/$ARM"
