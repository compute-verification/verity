# The node side of the TP drivers (tp_decode.sh, tp_replay.sh, tp_rerun.sh), sourced from the source tree's root on an 8-GPU H200 node:
# CASE's checkpoint (GLM53 by default, or DSV4_PRO) bound from the node's verified local copy (LOCAL_CK, a directory holding
# `.verified`; by default /workspace/cp/<model>-ckpt/<rev>) over the path the checkpoint manifest reads, the Hub offline, a scratch
# TMPDIR removed on exit, the venv (VENV, default /workspace/cp/venv312) and PYTHONPATH over the tree, and the working directory
# integrations/vllm. Sets TREE, the tree's root.
TREE=$PWD
case ${CASE:-GLM53} in
  DSV4_PRO) CK=dsv4 REV=b5968e9190ef611bbf34a7229255be88a0e937c1 ;;
  *) CK=glm53 REV=aca966e4e02791568aa6a4ced368624b3d897f42 ;;
esac
FS_CK=/workspace/fs/$CK/$REV
LOCAL_CK=${LOCAL_CK:-/workspace/cp/$CK-ckpt/$REV}
if [ -e "$LOCAL_CK/.verified" ] && ! mountpoint -q "$FS_CK"; then mkdir -p "$FS_CK" && sudo -n mount --bind "$LOCAL_CK" "$FS_CK"; fi
test -s "$FS_CK/config.json" || { echo "no checkpoint at $FS_CK (and no verified local copy at $LOCAL_CK)" >&2; exit 2; }
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export HF_HOME=/workspace/cp/hf HF_HUB_OFFLINE=1
TMPDIR=$(mktemp -d /tmp/tp-driver.XXXXXX)
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT
# shellcheck disable=SC1091
source "${VENV:-/workspace/cp/venv312}/bin/activate"
export PYTHONPATH=$TREE:$TREE/benchmarks:$TREE/integrations/vllm:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/benchmarks/numerical/python:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
cd "$TREE/integrations/vllm"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
