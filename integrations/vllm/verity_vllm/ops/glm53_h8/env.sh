# Sourced by every GLM-5.3 8xH200 step (`research run --cwd source`, so the cwd is the source tree's root).
# The engine environment of record: eager TP8 on sm_90, CUTLASS block FP8 (no DeepGEMM), no batch invariance (vLLM refuses it on
# sm_90), the shared expert on the main stream (#1267), the H200 tuned MoE configs of r20261012-220408-4943.
set -uo pipefail
O=$RESEARCH_RUN_DIR/out; mkdir -p "$O"
cd integrations/vllm || exit 3
V=$PWD
export PYTHONPATH=$V:$(cd ../.. && pwd -P):$(cd ../../tools/research/src && pwd -P)
export VERITY_TREE=$V HF_HOME=/workspace/cp/hf HF_HUB_OFFLINE=1
export PY=/workspace/cp/venv312/bin/python
export SWEEP_DIR=${SWEEP_DIR:-/workspace/cp/sweep} GUARD_PRESERVE_FILE=/workspace/cp/guard.preserve
REPO=zai-org/GLM-5.3 REV=aca966e4e02791568aa6a4ced368624b3d897f42 ROLE=GLM53
CK=/workspace/fs/glm53/$REV
ROW=glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
WL=workloads/$ROW.json
CMAP=manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json
mkdir -p /workspace/tmp "$SWEEP_DIR"
export TMPDIR=/workspace/tmp
# The bootstrap's torchcodec 0.17.0 is a CUDA 13 build (libnvrtc.so.13) beside torch cu129: `import vllm.multimodal` and cuteDSL's
# ll_bf16 raise OSError (r20261012-221745-18e0), and every Build derive with them (r20261012-234841-c3ca). It is replaced in the
# venv itself, since the row driver sets its children's PYTHONPATH (r20261012-235556-abc8; again here after a re-bootstrap).
TC=0.16.0+cu129
tc_ok() { [ "$("$PY" -c 'import importlib.metadata as m; print(m.version("torchcodec"))' 2>/dev/null)" = "$TC" ]; }
tc_ok || (flock 9; tc_ok || uv pip install --python "$PY" --no-deps --index-url https://download.pytorch.org/whl/cu129 \
  "torchcodec==$TC") 9> /workspace/cp/venv-torchcodec.lock >> "$O/venv-torchcodec.log" 2>&1
export VLLM_USE_DEEP_GEMM=0 VLLM_BATCH_INVARIANT=0 VLLM_TUNED_CONFIG_FOLDER=/workspace/fs/glm53-configs/moe_configs
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 NCCL_P2P_DISABLE=1
ENGINE_ARGS=(--engine-arg gpu_memory_utilization=0.75 --engine-arg max_model_len=288 --engine-arg max_num_batched_tokens=2048
             --engine-arg disable_custom_all_reduce=true)
VV() { "$PY" -m verity_vllm.pipeline.cli "$@"; }
stamp() { echo "[$1] $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg) mem_free_gib=$(free -g | awk '/Mem:/{print $7}')"; }
