#!/usr/bin/env bash
# J10-R's served H1 receipts on one GPU. h1u.cu's -hbc build (producer_gpu.sh's) and noise.cu's cubin, niced, then by JOB:
#   tests   the producer's GPU tests (tests/test_served_h1_producer.py -k "j10r or noise_cu or units_kernel": J10-R's and
#           v1's, the CPU J10-R tests with them) and j10r_time.py, J10-R's noise and receipts against v1's interleaved
#           (TIME_ARGS its extra flags), into $OUT/junit.xml and $OUT/j10r_time.json
#   window  a served Llama-3.1-8B int8 window over J10-R's tile unit with each call's receipt@<c> from the units kernel
#           (pouw_h1_gpu.py --workload fill --served --commit --negatives --receipts --j10r; DECLARE its --declare,
#           RECEIPTS=cubin noise.cu's torch chain in the units kernel's place, NOISE=v1 the window over v1's unit)
#
#   research run --on vy-nebius-2 --gpus 1 --mem-gb 96 --max-min 40 --cwd source --source . --env JOB=window \
#     -- bash benchmarks/pouw/served_h1/j10r_gpu.sh
set -euo pipefail
PY=${PY:-/workspace/jobs/venv312/bin/python}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
: "${OUT:?research run gives the run its OUT}"
ROOT=$(pwd)
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
nice -n 10 "$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a -Xcompiler -fPIC -shared \
  -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 -o "$S/h1u_hbc.so" benchmarks/hash/dither/h1u.cu \
  > "$OUT/build-hbc.log" 2>&1 || { echo "j10r_gpu: NVCC_FAIL h1u.cu"; tail -30 "$OUT/build-hbc.log"; exit 11; }
nice -n 10 "$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I benchmarks/codec_tensor -cubin \
  -o "$S/noise.cubin" benchmarks/pouw/served_h1/noise.cu > "$OUT/build-noise.log" 2>&1 \
  || { echo "j10r_gpu: NVCC_FAIL noise.cu"; tail -30 "$OUT/build-noise.log"; exit 11; }
sha256sum "$S/h1u_hbc.so" "$S/noise.cubin" | tee "$OUT/builds.txt"
export PYTHONPATH="$ROOT:$ROOT/integrations/vllm" HF_HOME=${HF_HOME:-/workspace/hf}
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv | tee "$OUT/gpu.txt"
case "${JOB:-tests}" in
  tests)
    cd benchmarks/pouw
    rc=0
    VERITY_POUW_H1U_SO="$S/h1u_hbc.so" VERITY_POUW_NOISE_CUBIN="$S/noise.cubin" \
      "$PY" -m pytest tests/test_served_h1_producer.py -k "j10r or noise_cu or units_kernel" -p no:cacheprovider -rA -v \
      --junitxml "$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log" || rc=1
    grep -q 'skipped="0"' "$OUT/junit.xml" && ! grep -q 'tests="0"' "$OUT/junit.xml" \
      || { echo "j10r_gpu: the tests didn't all run"; rc=1; }
    # shellcheck disable=SC2086
    "$PY" served_h1/j10r_time.py --so "$S/h1u_hbc.so" --cubin "$S/noise.cubin" --out "$OUT/j10r_time.json" ${TIME_ARGS:-} \
      2>&1 | tee "$OUT/j10r_time.log" || rc=1
    exit $rc ;;
  window)
    so="$S/h1u_hbc.so"
    [ "${RECEIPTS:-units}" = cubin ] && so="$S/noise.cubin"
    j10r=--j10r
    [ "${NOISE:-j10r}" = v1 ] && j10r=
    cd integrations/vllm
    # shellcheck disable=SC2086
    "$PY" -m tests.protocol_options.pouw_h1_gpu --workload fill --served --commit --negatives --receipts "$so" $j10r \
      --declare "${DECLARE:-32,3,32,4}" --out "$OUT" --at 2026-10-24T00:00:00Z --by compute-accounting 2>&1 \
      | tee "$OUT/window.log"
    exit "${PIPESTATUS[0]}" ;;
  *) echo "j10r_gpu: JOB is tests or window, not ${JOB}"; exit 2 ;;
esac
