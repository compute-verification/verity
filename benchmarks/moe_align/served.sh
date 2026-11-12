#!/usr/bin/env bash
# served.py: DeepSeek-V4-Pro TP8 eager with the canonical moe_align_block_size on its Marlin MoE path, from the source tree's
# root on an 8 x H200 node holding the checkpoint:
#   research run --on NODE --project verity --cwd source -- gpu-lease 8 --wait --mem-gb 64 -- bash benchmarks/moe_align/served.sh
# ROW (default DeepSeek-V4-Pro's TP8 eager B1, whose request every batch rotates), CHECK_BATCHES (default 1,8), BATCHES (default
# 1,16,64,256), PAIRS (default 3), TP (default 8), VENV (default /workspace/cp/venv312), GPU_UTIL (default 0.92), ENGINE_ARGS
# (space-separated K=V, each one more --engine-arg, e.g. max_num_batched_tokens=16384 under the largest batch's), VARY_REPEATS
# (default 0: each op's every call repeated N more times on its inputs, at VARY_BATCHES, default CHECK_BATCHES). LOCAL_CK: a
# verified copy of the checkpoint (it holds .verified), bind-mounted on the path manifests/checkpoints.json names when nothing is
# mounted there. The integration's vLLM plugin loads from a site directory declaring its entry points (stress.py site).
# DUMMY_LAYERS=N: no checkpoint; the first N layers with dummy weights (served.py --dummy-layers), e.g. TP2 on a node without it:
#   research run --on NODE --project verity --cwd source -- gpu-lease 2 --wait -- env TP=2 DUMMY_LAYERS=5 bash benchmarks/moe_align/served.sh
# Out: $OUT (research run sets it): versions.json, check.json, decode.jsonl, host.json, summary.json.
set -euo pipefail
TREE=$PWD
OUT=${OUT:?set OUT}
ROW=${ROW:-dsv4-pro__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager}
FS_CK=/workspace/fs/dsv4/b5968e9190ef611bbf34a7229255be88a0e937c1
if [ -z "${DUMMY_LAYERS:-}" ]; then
  if [ -n "${LOCAL_CK:-}" ] && [ -e "$LOCAL_CK/.verified" ] && ! mountpoint -q "$FS_CK"; then
    sudo -n mkdir -p "$FS_CK" && sudo -n mount --bind "$LOCAL_CK" "$FS_CK"
  fi
  [ -e "$FS_CK/config.json" ] || { echo "no checkpoint at $FS_CK (LOCAL_CK names a verified copy to mount there)"; exit 3; }
fi
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM VLLM_PLUGINS
export HF_HOME=/workspace/cp/hf HF_HUB_OFFLINE=1
TMPDIR=$(mktemp -d /tmp/moe-align-served.XXXXXX)
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT
# shellcheck disable=SC1091
source "${VENV:-/workspace/cp/venv312}/bin/activate"
export PYTHONPATH=$TREE:$TREE/benchmarks:$TREE/adapters/vllm:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/benchmarks/numerical/python:$TREE/verity/protocols/flock/prover/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/difftest/python
python "$TREE/benchmarks/moe_align/stress.py" site --out "$TMPDIR/site" > /dev/null
export PYTHONPATH=$TMPDIR/site:$PYTHONPATH
mkdir -p "$OUT"
cd "$TREE/adapters/vllm"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader | tee "$OUT/gpus.csv"
extra=()
for kv in ${ENGINE_ARGS:-}; do extra+=(--engine-arg "$kv"); done
python -u -m moe_align.served --workload "workloads/$ROW.json" --tp "${TP:-8}" --check-batches "${CHECK_BATCHES:-1,8}" \
  --batches "${BATCHES:-1,16,64,256}" --pairs "${PAIRS:-3}" --engine-arg "gpu_memory_utilization=${GPU_UTIL:-0.92}" \
  "${extra[@]}" --dummy-layers "${DUMMY_LAYERS:-0}" --vary-repeats "${VARY_REPEATS:-0}" \
  --vary-batches "${VARY_BATCHES:-${CHECK_BATCHES:-1,8}}" --out "$OUT"
