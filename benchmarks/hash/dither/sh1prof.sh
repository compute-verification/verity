#!/usr/bin/env bash
# SERVE-H1's served step, phase by phase (sh1prof.py around serve.py, unchanged), under nsys on one leased GPU for at most
# 28 minutes, then nsys's NVTX, NVTX-GPU-projection and kernel summaries.  Extra arguments go to serve.py.
#   research run --on vy-nebius-2 --project verity --campaign dither-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-dither-rtx -- bash benchmarks/hash/dither/sh1prof.sh --max-tokens 4 --coins live --skip-timing
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-dither-sh1.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
NSYS=${NSYS:-$CUDA/bin/nsys}
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I benchmarks/codec_tensor -cubin \
  -o "$S/noise.cubin" benchmarks/pouw/served_h1/noise.cu > "$OUT/nvcc.txt" 2>&1 \
  || { echo "sh1prof: NVCC_FAIL"; tail -20 "$OUT/nvcc.txt"; exit 11; }
cd integrations/vllm
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/benchmarks/pouw/served_h1:$ROOT/benchmarks/hash/dither
export HF_HOME=${HF_HOME:-/workspace/hf} FLASHINFER_CUDA_ARCH_LIST=12.0f
echo "sh1prof: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --max-min 28 -- timeout 1650 "$NSYS" profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o "$S/sh1" --force-overwrite true "$PY" "$ROOT/benchmarks/hash/dither/sh1prof.py" --phases "$OUT/phases.json" -- \
  --out "$OUT/serve" --cubin "$S/noise.cubin" --scratch "$S" --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --by kernels "$@"
rc=$?
echo "sh1prof: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/sh1.nsys-rep" ]; then
  "$NSYS" stats --report nvtx_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum --format csv -o "$OUT/sh1" "$S/sh1.nsys-rep" \
    > "$OUT/stats.log" 2>&1 || echo "sh1prof: nsys stats rc $?"
  ls -la "$S/sh1.nsys-rep" | tee "$OUT/rep-size.txt"
fi
exit "$rc"
