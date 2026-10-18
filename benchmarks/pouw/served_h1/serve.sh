#!/usr/bin/env bash
# served_h1's window on node 2 (serve.py): noise.cu built for sm_120a on the CPU, then serve.py on one leased GPU for at most
# 28 minutes.  Extra arguments go to serve.py.
#   research run --on vy-nebius-2 --project verity --campaign hash-loop --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=<id> -- bash benchmarks/pouw/served_h1/serve.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-served-h1.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/tree.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I benchmarks/codec_tensor -cubin \
  -o "$S/noise.cubin" benchmarks/pouw/served_h1/noise.cu > "$OUT/nvcc.txt" 2>&1 \
  || { echo "served-h1: NVCC_FAIL"; tail -20 "$OUT/nvcc.txt"; exit 11; }
cd integrations/vllm
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/benchmarks/pouw/served_h1
export HF_HOME=${HF_HOME:-/workspace/hf} FLASHINFER_CUDA_ARCH_LIST=12.0f
echo "served-h1: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --max-min 28 -- timeout 1650 "$PY" "$ROOT/benchmarks/pouw/served_h1/serve.py" --out "$OUT" \
  --cubin "$S/noise.cubin" --scratch "$S" --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --by compute-accounting "$@"
rc=$?
echo "served-h1: serve.py exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
