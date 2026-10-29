#!/usr/bin/env bash
# FA3's knobs for the served prefill attention on the H200 (attn_knobs.py), under `research run` (cwd: the root of a
# source tree):
#
#   research run --on MACHINE --project verity --source . --cwd source --timeout 40m \
#     --env GPU_LEASE_WHO=LANE [--env VENV=DIR] [--env FA3_SO=NS=PATH] -- bash benchmarks/pouw/gemm_rate/attn_knobs.sh
#
# Outside the lease: stamp.cu's cubin, and the served vLLM wheel (attn_rate.sh's WHEEL) in VENV (default the run's work
# directory) unless VENV already has it. Under one gpu-lease (MAX_MIN, default 12): HARNESS.py (default attn_knobs, or
# attn_kfa3 for the gated not-byte-equal arms) over PROMPTS, ARMS, each FA3_SO, WINDOW_MS, WINDOW_LEAD_MS and WINDOW_REPS,
# with nvidia-smi sampling the SM clock. $OUT gets knobs.jsonl, table.md, smi.csv and rc.txt.
set -uo pipefail
export O=${OUT:?}
TREE=$PWD
E=benchmarks/pouw/gemm_rate
CUDA=${CUDA:-/usr/local/cuda-13.0}
export W=${RESEARCH_RUN_DIR:-$O}/work
mkdir -p "$W"
export XDG_CACHE_HOME=$W/cache TRITON_CACHE_DIR=$W/triton
WHEEL=${WHEEL:-https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}
VENV=${VENV:-$W/venv}
PROMPTS=${PROMPTS:-"8x1024 16x1024 32x1024"}
export HARNESS=${HARNESS:-attn_knobs}
export ARMS=${ARMS:-served,served-eager,kattn,sched,contig,contig-sched,page128,splits0}
export WINDOW_MS=${WINDOW_MS:-2000} WINDOW_LEAD_MS=${WINDOW_LEAD_MS:-500} WINDOW_REPS=${WINDOW_REPS:-3}
git rev-parse HEAD > "$O/commit.txt" 2>&1
: > "$O/rc.txt"
"$CUDA/bin/nvcc" -O3 -gencode arch=compute_90a,code=sm_90a -cubin -o "$W/stamp.cubin" $E/stamp.cu > "$O/build-stamp.log" 2>&1
echo "build stamp rc $?" >> "$O/rc.txt"
if ! "$VENV/bin/python" -c "import vllm.vllm_flash_attn" > /dev/null 2>&1; then
  if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$W/bin" UV_NO_MODIFY_PATH=1 sh > "$O/uv-install.log" 2>&1
    export PATH=$W/bin:$PATH
  fi
  uv venv -q --python 3.12 "$VENV" > "$O/venv.log" 2>&1 \
    && uv pip install -q --python "$VENV/bin/python" "$WHEEL" --extra-index-url "$TORCH_INDEX" \
      --index-strategy unsafe-best-match >> "$O/venv.log" 2>&1
  echo "venv rc $?" >> "$O/rc.txt"
fi
export PY=$VENV/bin/python PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/config
$PY -c "import importlib.metadata as md, torch; print({p: md.version(p) for p in ('vllm', 'torch')}, torch.version.cuda)" \
  > "$O/versions.txt" 2>&1
echo "imports rc $?" >> "$O/rc.txt"
grep -qv ' rc 0$' "$O/rc.txt" && { cat "$O/rc.txt"; tail -30 "$O/venv.log" "$O/versions.txt" 2> /dev/null; exit 1; }

a=()
for p in $PROMPTS; do a+=(--prompt "$p"); done
for s in ${FA3_SO:-}; do a+=(--fa3-so "$s"); done
LEASE=${LEASE-gpu-lease 1 --wait --max-min ${MAX_MIN:-12} --}
# shellcheck disable=SC2016
$LEASE bash -c '
  G=(); [ -n "${GPU_LEASE_UUID:-}" ] && G=(-i "$GPU_LEASE_UUID")
  nvidia-smi "${G[@]}" --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 > "$O/smi.csv" 2>&1 &
  smi=$!
  "$PY" "benchmarks/pouw/gemm_rate/$HARNESS.py" --stamp-cubin "$W/stamp.cubin" "$@" --arms "$ARMS" \
    --window-ms "$WINDOW_MS" --lead-ms "$WINDOW_LEAD_MS" --rounds "$WINDOW_REPS" --table "$O/table.md" \
    > "$O/knobs.jsonl" 2> "$O/knobs.err"
  echo "knobs rc $?" >> "$O/rc.txt"
  kill $smi' _ "${a[@]}"
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt" "$O/table.md" 2> /dev/null
! grep -qv ' rc 0$' "$O/rc.txt"
