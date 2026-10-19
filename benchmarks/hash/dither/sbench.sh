#!/usr/bin/env bash
# Dither's served standard number on node 2 (sbench.py), on one leased GPU for at most 40 minutes.  Extra arguments go to
# sbench.py (e.g. --arms stock,exact-eager --reps 3).
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
if [ -f benchmarks/hash/dither/arm.sh ]; then bash benchmarks/hash/dither/arm.sh "$S" > "$OUT/arm-build.log" 2>&1 || { echo "sbench: ARM_BUILD_FAIL"; tail -30 "$OUT/arm-build.log"; exit 11; }; fi
cd integrations/vllm
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/benchmarks/hash/dither
export HF_HOME=${HF_HOME:-/workspace/hf} FLASHINFER_CUDA_ARCH_LIST=12.0f DITHER_BUILD=$S
echo "sbench: asking for one GPU at $(date -u +%H:%M:%SZ)"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 2 > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
gpu-lease 1 --wait --max-min 40 -- timeout 2350 "$PY" "$ROOT/benchmarks/hash/dither/sbench.py" --out "$OUT" --scratch "$S" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "sbench: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
