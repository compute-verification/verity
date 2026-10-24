#!/usr/bin/env bash
# A served Llama-3.1-8B int8 H1 window with each call's receipt@<c> from the producer on the serving GPU
# (pouw_h1_gpu.py --workload fill --served --commit --receipts): h1u.cu's -hbc build (producer_gpu.sh's), niced, before the
# lease, then one GPU for the window and its Commit window, the Commit window held to core's ROOTS1 steps (core_verdict)
# and with one receipt bit flipped (receipt_tamper, receipt_statement). DECLARE is --declare's SLOTS,CAP,T,S, LEASE_MIN
# the lease's minutes, GPU_LEASE_ON, if set, gpu-lease's --on.
#
#   research run --on vy-nebius-2 --cwd source --env GPU_LEASE_ON=2,3 [--env DECLARE=64,3,64,4 --env LEASE_MIN=30] \
#     -- bash benchmarks/pouw/served_h1/window_gpu.sh
set -euo pipefail
PY=${PY:-/workspace/jobs/venv312/bin/python}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
: "${OUT:?research run gives the run its OUT}"
ROOT=$(pwd)
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
nice -n 10 "$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared \
  -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 -o "$S/h1u_hbc.so" benchmarks/hash/dither/h1u.cu \
  > "$OUT/build-hbc.log" 2>&1 || { echo "window_gpu: NVCC_FAIL"; tail -30 "$OUT/build-hbc.log"; exit 11; }
sha256sum "$S/h1u_hbc.so" | tee "$OUT/builds.txt"
export PYTHONPATH="$ROOT:$ROOT/integrations/vllm" HF_HOME=${HF_HOME:-/workspace/hf} GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels-rcpt}
cd integrations/vllm
gpu-lease 1 --wait ${GPU_LEASE_ON:+--on "$GPU_LEASE_ON"} --max-min "${LEASE_MIN:-20}" --mem-gb "${MEM_GB:-96}" -- \
  "$PY" -m tests.protocol_options.pouw_h1_gpu --workload fill --served --commit --receipts "$S/h1u_hbc.so" \
  --declare "${DECLARE:-32,3,32,4}" --out "$OUT" --at 2026-10-21T00:00:00Z --by kernels-rcpt 2>&1 | tee "$OUT/window.log"
