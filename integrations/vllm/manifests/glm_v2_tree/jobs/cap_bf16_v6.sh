#!/usr/bin/env bash
# The bf16 v6 capture (`glm53_bf16_gemm_capture`, verity-vllm/glm53-bf16-gemm-capture/v6) on one H200, the one conformance capture
# the v2 rows need that the store lacks (#2838: the other eight served-default captures stand).  The harness runs the split-K-off
# variant and then vLLM's served default in a second process, reading the configuration back before and after.
#   research run --on vy-glm53-eu --gpus 1 --mem-gb 64 --timeout 45m --cwd source --project verity -- bash integrations/vllm/manifests/glm_v2_tree/jobs/cap_bf16_v6.sh
# then, from any VM: `verity-vllm records capture RUN --property glm53_bf16_gemm_capture` prints the art.
set -uo pipefail
TREE=$PWD
REV=aca966e4e02791568aa6a4ced368624b3d897f42
FS_CK=/workspace/fs/glm53/$REV; LOCAL_CK=/workspace/cp/glm53-ckpt/$REV
if [ -e "$LOCAL_CK/.verified" ] && ! mountpoint -q "$FS_CK"; then sudo -n mount --bind "$LOCAL_CK" "$FS_CK"; fi
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee "$OUT/gpus_at_start.txt"
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM VLLM_BATCH_INVARIANT
source /workspace/cp/venv312/bin/activate
export HF_HOME=/workspace/cp/hf HF_HUB_OFFLINE=1 PATH=$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src
W=$(mktemp -d /tmp/glm-v2-bf16.XXXXXX); trap 'rm -rf "$W"' EXIT
cd "$TREE/integrations/vllm"
python -u tests/properties/glm53_bf16_gemm_capture_gpu.py --out "$OUT/bf16_v6" --scratch "$W" 2>&1 | tee "$OUT/bf16_v6.out"
rc=${PIPESTATUS[0]}
grep -E '^GLM53-BF16-GEMM (GO|NO-GO)' "$OUT/bf16_v6.out" | tail -1
exit "$rc"
