#!/usr/bin/env bash
# h1r.cu built for sm_120a (and served_h1's noise.cu, receipts.py's reference noise) on the CPU, then h1rgate.py on one leased GPU for
# at most 30 minutes: the gate on a full served step, then the timing.  Extra arguments go to h1rgate.py.
# On an RTX PRO 6000 under a GPU lease:
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/h1r.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-dither-h1r.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -Xcompiler -fPIC -shared \
  -o "$S/libh1r.so" benchmarks/hash/dither/h1r.cu > "$OUT/build.log" 2>&1 \
  || { echo "h1r: NVCC_FAIL"; grep -v "^ptxas info" "$OUT/build.log" | tail -30; exit 11; }
grep -A2 "h1r_units_k\|h1r_seeds" "$OUT/build.log" | grep -i "registers\|spill" | head
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin \
  -o "$S/noise.cubin" benchmarks/pouw/served_h1/noise.cu > "$OUT/nvcc-noise.txt" 2>&1 \
  || { echo "h1r: NVCC_FAIL noise.cu"; tail -20 "$OUT/nvcc-noise.txt"; exit 11; }
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PYTHONPATH=$ROOT:$ROOT/benchmarks/pouw/served_h1:$ROOT/benchmarks/hash/dither
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
echo "h1r: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --max-min 30 -- timeout 1750 "$PY" benchmarks/hash/dither/h1rgate.py --lib "$S/libh1r.so" --cubin "$S/noise.cubin" \
  --out "$OUT" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "h1r: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
