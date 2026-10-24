#!/usr/bin/env bash
# The receipt producer's GPU test (tests/test_served_h1_producer.py's units-kernel test: every call's receipt@<c> through
# vllm's Served at one Llama-3.1-8B layer, against receipts.py's torch chain and h1tile.reference) for each -hbc build of
# h1u.cu: sbench.sh's dither-a-hbc flags alone, and with #3290's levers (H1U_PF, H1U_NB2, H1U_L2PF). The builds run before
# the lease; each test holds one GPU, about 5 minutes. GPU_LEASE_ON, if set, is gpu-lease's --on: the GPUs it may take.
#
#   research run --on vy-nebius-2 --cwd source [--env GPU_LEASE_ON=0,2,3] -- bash benchmarks/pouw/served_h1/producer_gpu.sh
set -euo pipefail
PY=${PY:-/workspace/jobs/venv312/bin/python}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
: "${OUT:?research run gives the run its OUT}"
ROOT=$(pwd)
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
declare -A LEVERS=([hbc]="" [hbc-3290]="-DH1U_PF=1 -DH1U_NB2=1 -DH1U_L2PF=1")
for name in hbc hbc-3290; do
  read -r -a kc <<< "${LEVERS[$name]}"
  "$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared \
    -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 "${kc[@]}" -o "$S/h1u_$name.so" benchmarks/hash/dither/h1u.cu \
    > "$OUT/build-$name.log" 2>&1 || { echo "producer_gpu: NVCC_FAIL $name"; tail -30 "$OUT/build-$name.log"; exit 11; }
  sha256sum "$S/h1u_$name.so" | tee -a "$OUT/builds.txt"
done
export PYTHONPATH="$ROOT:$ROOT/integrations/vllm" GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels-rcpt}
cd benchmarks/pouw
rc=0
for name in hbc hbc-3290; do
  VERITY_POUW_H1U_SO="$S/h1u_$name.so" gpu-lease 1 --wait ${GPU_LEASE_ON:+--on "$GPU_LEASE_ON"} --max-min 20 -- \
    "$PY" -m pytest tests/test_served_h1_producer.py -k units_kernel -p no:cacheprovider -rA -v \
    --junitxml "$OUT/junit-$name.xml" 2>&1 | tee "$OUT/pytest-$name.log" || rc=1
  grep -q 'tests="1"[^>]*skipped="0"\|skipped="0"[^>]*tests="1"' "$OUT/junit-$name.xml" \
    || { echo "producer_gpu: $name's test didn't run"; rc=1; }
done
exit $rc
