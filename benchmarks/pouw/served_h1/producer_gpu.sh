#!/usr/bin/env bash
# The receipt producer's GPU tests (tests/test_served_h1_producer.py's units-kernel tests: every call's receipt@<c> through
# vllm's Served at one Llama-3.1-8B layer, against receipts.py's torch chain and h1tile.reference, the kernel with dead
# tiles skipped against h1tile.reference's v4, a step's calls of one K in one launch against each call's own, and J10-R's:
# noise.cu's call streams against hashlib's and the units kernel's J10-R receipts against h1tile.j10r_reference) for each
# -hbc build of h1u.cu in BUILDS: hbc, sbench.sh's dither-a-hbc flags alone, and hbc-3290, with its levers (H1U_PF,
# H1U_NB2, H1U_L2PF); then, under the same lease, the producer's time a decode step with dead tiles computed and skipped,
# a call at a time and batched, at each of STEP_ROWS (comma-separated, of 64) served rows (step_time.py; empty: none).
# The builds run before the lease, niced; each build's tests and timing hold one GPU, about 10 minutes. GPU_LEASE_ON, if
# set, is gpu-lease's --on: the GPUs it may take; LEASE_MIN its --max-min (20).
#
#   research run --on vy-nebius-2 --cwd source --source . [--env GPU_LEASE_ON=2,3] [--env BUILDS=hbc] \
#     -- bash benchmarks/pouw/served_h1/producer_gpu.sh
set -euo pipefail
PY=${PY:-/workspace/jobs/venv312/bin/python}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
: "${OUT:?research run gives the run its OUT}"
ROOT=$(pwd)
. "$(dirname "$0")/arch.sh"
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
read -r -a builds <<< "${BUILDS:-hbc hbc-3290}"
declare -A LEVERS=([hbc]="" [hbc-3290]="-DH1U_PF=1 -DH1U_NB2=1 -DH1U_L2PF=1")
for name in "${builds[@]}"; do
  [ -n "${LEVERS[$name]+x}" ] || { echo "producer_gpu: no build $name (${!LEVERS[*]})"; exit 2; }
  read -r -a kc <<< "${LEVERS[$name]}"
  nice -n 10 "$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr -gencode "$GENCODE" -Xcompiler -fPIC \
    -shared -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 "${kc[@]}" -o "$S/h1u_$name.so" benchmarks/hash/dither/h1u.cu \
    > "$OUT/build-$name.log" 2>&1 || { echo "producer_gpu: NVCC_FAIL $name"; tail -30 "$OUT/build-$name.log"; exit 11; }
  sha256sum "$S/h1u_$name.so" | tee -a "$OUT/builds.txt"
done
nice -n 10 "$NVCC" -O3 -std=c++17 -gencode "$GENCODE" -I benchmarks/codec_tensor -cubin \
  -o "$S/noise.cubin" benchmarks/pouw/served_h1/noise.cu > "$OUT/build-noise.log" 2>&1 \
  || { echo "producer_gpu: NVCC_FAIL noise.cu"; tail -30 "$OUT/build-noise.log"; exit 11; }
sha256sum "$S/noise.cubin" | tee -a "$OUT/builds.txt"
export VERITY_POUW_NOISE_CUBIN="$S/noise.cubin"
export PYTHONPATH="$ROOT:$ROOT/tools/config:$ROOT/integrations/vllm" GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels-rcpt}
cd benchmarks/pouw
rc=0
for name in "${builds[@]}"; do
  VERITY_POUW_H1U_SO="$S/h1u_$name.so" gpu-lease 1 --wait ${GPU_LEASE_ON:+--on "$GPU_LEASE_ON"} --max-min "${LEASE_MIN:-20}" -- \
    bash -c '"$1" -m pytest tests/test_served_h1_producer.py -k "units_kernel or noise_cu" -p no:cacheprovider -rA -v --junitxml "$2"
             t=$?
             [ -z "$3" ] || "$1" served_h1/step_time.py --so "$VERITY_POUW_H1U_SO" --rows "$3" --out "$4" || t=1
             exit $t' _ "$PY" "$OUT/junit-$name.xml" "${STEP_ROWS-20,64}" "$OUT/step-$name.json" \
    2>&1 | tee "$OUT/pytest-$name.log" || rc=1
  grep -q 'skipped="0"' "$OUT/junit-$name.xml" && ! grep -q 'tests="0"' "$OUT/junit-$name.xml" \
    || { echo "producer_gpu: $name's tests didn't all run"; rc=1; }
done
exit $rc
