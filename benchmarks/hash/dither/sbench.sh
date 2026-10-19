#!/usr/bin/env bash
# Dither's served standard number on node 2 (sbench.py), on one leased GPU for at most 40 minutes: libh1c.so and libh1f.so
# built for sm_120a first (the dither arm's).  Extra arguments go to sbench.py (e.g. --arms exact,dither --gate --ops decode
# --t 64).  DITHER_NSYS=1 runs it under nsys, recording the kernels inside the gate's CUDA profiler range (each graph's
# kernels as nodes), then their summary.
#   research run --on vy-nebius-2 --project verity --campaign dither-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-dither-rtx -- bash benchmarks/hash/dither/sbench.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-dither-sbench.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$CUDA/bin:$PATH
for lib in h1c h1f; do
  nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -Xcompiler -fPIC -shared \
    -o "$S/lib$lib.so" "benchmarks/hash/dither/$lib.cu" > "$OUT/build-$lib.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL $lib"; grep -v "^ptxas info" "$OUT/build-$lib.log" | tail -30; exit 11; }
done
grep -B1 -A2 "h1f_k\|h1r_units" "$OUT/build-h1f.log" | grep -i "registers\|spill" | head -6
export DITHER_H1C=$S/libh1c.so DITHER_H1F=$S/libh1f.so VLLM_DISABLE_COMPILE_CACHE=1
cd integrations/vllm
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PATH=$(dirname "$PY"):$PATH
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/benchmarks/pouw/served_h1:$ROOT/benchmarks/hash/dither
export HF_HOME=${HF_HOME:-/workspace/hf} FLASHINFER_CUDA_ARCH_LIST=12.0f
NSYS=()
if [ "${DITHER_NSYS:-}" = 1 ]; then
  NSYS=("$CUDA/bin/nsys" profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/gate" --force-overwrite true)
fi
echo "sbench: asking for one GPU at $(date -u +%H:%M:%SZ)"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 2 > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
gpu-lease 1 --wait --max-min 40 -- timeout 2350 "${NSYS[@]}" "$PY" "$ROOT/benchmarks/hash/dither/sbench.py" --out "$OUT" --scratch "$S" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "sbench: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/gate.nsys-rep" ]; then
  "$CUDA/bin/nsys" stats --report cuda_gpu_kern_sum --format csv -o "$OUT/gate" "$S/gate.nsys-rep" > "$OUT/nsys-stats.log" 2>&1 \
    || echo "sbench: nsys stats rc $?"
  "$PY" "$ROOT/benchmarks/hash/dither/kernels.py" "$OUT"/gate_cuda_gpu_kern_sum.csv | tee "$OUT/kernels.json"
fi
exit "$rc"
