#!/usr/bin/env bash
# Sampled proofs' Commit against plain FP8 serving through the shared vLLM benchmark (`verity-bench vllm`), Llama-3.1-8B on
# one GPU, eager (cwd = the source tree).  Builds the collector's three JIT extensions outside the lease, then runs the driver
# under `gpu-lease 1 --wait`.  Results go to $RESEARCH_RUN_DIR/out/bench.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 100 --campaign proofs \
#     --env GPU_LEASE_WHO=proofs-sp-arm --env SWEEP=quick -- bash benchmarks/sampled_proofs_commit/run.sh
#
# SWEEP (quick | full, default quick), REPS (default 3), MAXMIN (the lease's bound in minutes, default 60).  WORK is the
# e2e's vLLM environment on node 2 (benchmarks/pouw/pearl_c_vllm/setup.sh) and MODEL its Llama-3.1-8B-Instruct.
# MODE=build stops after the extensions are built.  --mem-gb covers the arm's pinned pool, capped at VERITY_SP_ARM_HOST_GB (80).
set -euo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || { echo "no $WORK/ready.json: run benchmarks/pouw/pearl_c_vllm/setup.sh"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export PATH=$CUDA_HOME/bin:$WORK/venv312/bin:$PATH
export PYTHONPATH=$PWD/benchmarks/harness:$PWD/integrations/vllm:$PWD/tools/research/src:$PWD
"$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps({k: d.get(k) for k in ("model", "revision", "vllm", "torch")}))' \
  "$WORK/ready.json" > "$O/ready.json"
"$PY" - > "$O/build.json" <<'EOF'
import json, os, sys, time
from verity_vllm.commit.committer import leafhash, native_collect
sys.path.insert(0, os.path.join(os.path.dirname(native_collect.__file__), "hidden_gpu_src"))
import hidden_gpu
out = {}
for name, ext in (("native_collect", native_collect.ext), ("hidden_gpu_tree", hidden_gpu.ext), ("leafhash", leafhash.ext)):
    t0 = time.perf_counter()
    ext()
    out[name] = round(time.perf_counter() - t0, 1)
print(json.dumps(out))
EOF
cat "$O/build.json"
[ "${MODE:-}" = build ] && exit 0
cat "$MODEL"/*.safetensors > /dev/null
gpu-lease 1 --wait --max-min "${MAXMIN:-60}" -- bash -c '
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  nvidia-smi --query-gpu=index,uuid,name,clocks.max.sm,memory.used --format=csv,noheader -i "${GPU_LEASE_UUID%%,*}" > "'"$O"'/gpu.txt"
  "'"$PY"'" -m verity_bench.cli vllm --preset llama31-8b-fp8-1gpu --model-path "'"$MODEL"'" --eager \
    --arms stock-fp8,sp-commit --arm-module benchmarks.sampled_proofs_commit.bench_arm:COMMIT \
    --sweep "'"${SWEEP:-quick}"'" --reps "'"${REPS:-3}"'" --out "'"$O"'/bench"'
