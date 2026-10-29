#!/usr/bin/env bash
# Prefill attention's MACs per SM clock on the H200 against bf16's 2,048, with the served int8 linears beside it
# (attn_rate.py), under `research run` (cwd: the root of a source tree):
#
#   research run --on MACHINE --project verity --campaign CAMPAIGN --source . --cwd source --timeout 90m \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/gemm_rate/attn_rate.sh
#
# Outside the lease: stamp.cu's cubin and attn_bf16's (build.sh, CUBIN_ONLY), and, unless PY names a Python with the served
# vLLM, a venv in the run directory with the served vLLM wheel (WHEEL) and its torch (TORCH_INDEX); then an import of
# every arm's module. Under one gpu-lease (LEASE, default gpu-lease 1 --wait --max-min MAX_MIN --, MAX_MIN 20):
# attn_rate.py over PROMPTS at CHUNK (8192), ARMS, WINDOW_MS (2000), WINDOW_LEAD_MS (500) and WINDOW_REPS (3), with
# nvidia-smi sampling the SM clock throughout (smi.csv). $OUT gets attn.jsonl, table.md, versions.txt and rc.txt.
set -uo pipefail
export O=${OUT:?}
TREE=$PWD
E=benchmarks/pouw/gemm_rate
CUDA=${CUDA:-/usr/local/cuda-13.0}
export W=${RESEARCH_RUN_DIR:-$O}/work
mkdir -p "$W"
export XDG_CACHE_HOME=$W/cache TRITON_CACHE_DIR=$W/triton UV_CACHE_DIR=$W/uv-cache
WHEEL=${WHEEL:-https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}
PROMPTS=${PROMPTS:-"1024 4096 8192 16384 8x1024 16x1024 32x1024"}
export CHUNK=${CHUNK:-8192} ARMS=${ARMS:-fa3,fa3-eager,fa2,bf16,lin}
export WINDOW_MS=${WINDOW_MS:-2000} WINDOW_LEAD_MS=${WINDOW_LEAD_MS:-500} WINDOW_REPS=${WINDOW_REPS:-3}
git rev-parse HEAD > "$O/commit.txt" 2>&1
gpu-lease status > "$O/lease-status.txt" 2>&1
: > "$O/rc.txt"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_90a,code=sm_90a -cubin -o "$W/stamp.cubin" $E/stamp.cu > "$O/build-stamp.log" 2>&1
echo "build stamp rc $?" >> "$O/rc.txt"
CUDA=$CUDA CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_bf16/build.sh "$W/attn_bf16" > "$O/build-attn_bf16.log" 2>&1
echo "build attn_bf16 rc $?" >> "$O/rc.txt"
if [ -z "${PY:-}" ]; then
  if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$W/bin" UV_NO_MODIFY_PATH=1 sh > "$O/uv-install.log" 2>&1
    export PATH=$W/bin:$PATH
  fi
  uv venv -q --python 3.12 "$W/venv" > "$O/venv.log" 2>&1 \
    && uv pip install -q --python "$W/venv/bin/python" "$WHEEL" --extra-index-url "$TORCH_INDEX" \
      --index-strategy unsafe-best-match >> "$O/venv.log" 2>&1
  echo "venv rc $?" >> "$O/rc.txt"
  PY=$W/venv/bin/python
fi
export PY PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/config
export ATTN_BF16_CUBIN=$W/attn_bf16/attn_bf16.sm_90a.cubin
$PY - > "$O/versions.txt" 2>&1 << 'EOF'
import importlib.metadata as md
import torch
from vllm.vllm_flash_attn import flash_attn_interface  # noqa: F401
from verity_vllm.engine import int8_fused  # noqa: F401
print({p: md.version(p) for p in ("vllm", "torch", "triton")}, "cuda", torch.version.cuda)
EOF
echo "imports rc $?" >> "$O/rc.txt"
grep -qv ' rc 0$' "$O/rc.txt" && { cat "$O/rc.txt"; tail -30 "$O/venv.log" "$O/versions.txt" 2> /dev/null; exit 1; }

a=()
for p in $PROMPTS; do a+=(--prompt "$p"); done
LEASE=${LEASE-gpu-lease 1 --wait --max-min ${MAX_MIN:-20} --}
# shellcheck disable=SC2016
$LEASE bash -c '
  G=(); [ -n "${GPU_LEASE_UUID:-}" ] && G=(-i "$GPU_LEASE_UUID")
  nvidia-smi "${G[@]}" --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,memory.used \
    --format=csv > "$O/gpu.txt" 2>&1
  nvidia-smi "${G[@]}" --query-gpu=timestamp,clocks.sm,temperature.gpu --format=csv -lms 500 > "$O/smi.csv" 2>&1 &
  smi=$!
  "$PY" benchmarks/pouw/gemm_rate/attn_rate.py --stamp-cubin "$W/stamp.cubin" "$@" --chunk "$CHUNK" --arms "$ARMS" \
    --window-ms "$WINDOW_MS" --lead-ms "$WINDOW_LEAD_MS" --rounds "$WINDOW_REPS" --table "$O/table.md" \
    > "$O/attn.jsonl" 2> "$O/attn.err"
  echo "attn rc $?" >> "$O/rc.txt"
  kill $smi' _ "${a[@]}"
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt" "$O/table.md" 2> /dev/null
! grep -qv ' rc 0$' "$O/rc.txt"
