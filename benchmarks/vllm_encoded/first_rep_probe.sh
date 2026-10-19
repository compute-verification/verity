#!/usr/bin/env bash
# MEM-41's probe (temporary): `bash benchmarks/vllm_encoded/first_rep_probe.sh [ARM...]` from the source tree inside a
# 1-GPU lease, as run_point.sh runs a point of the llama31-8b-fp8-1gpu-bandwidth spec (its env, and its library setup),
# but one engine lifetime per arm with every rep's steps recorded (first_rep_probe.py) into $OUT.
set -euo pipefail
WORK=${WORK:-/workspace}
VENV=${VERITY_BENCH_VENV:-$WORK/venv312}
export HF_HOME=${HF_HOME:-$WORK/hf}
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -d "$c" ] && { export CUDA_HOME=$c; break; }; done
fi
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$VENV/bin:$PATH
ROOT=$PWD
export PYTHONPATH=$ROOT/benchmarks/harness:$ROOT/integrations/vllm:$ROOT/tools/config:$ROOT/tools/research/src:$ROOT
export FLASHINFER_CUDA_ARCH_LIST=12.0f VERITY_VLLM_BENCH_KV_RECEIPTS=on
OUT=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}
mkdir -p "$OUT"
# shellcheck source=benchmarks/vllm_encoded/first_rep_setup.sh
source benchmarks/vllm_encoded/first_rep_setup.sh
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv > "$OUT/gpu.csv" || true
for arm in "${@:-plain audit}"; do
  for a in $arm; do
    "$VENV/bin/python" -m benchmarks.vllm_encoded.first_rep_probe --arm "$a" --out "$OUT"
  done
done
