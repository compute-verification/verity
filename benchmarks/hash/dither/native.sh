#!/usr/bin/env bash
# The native baseline on one H200 or B200: native.py's arms, each in its own process under its own gpu-lease (an engine build
# plus its measures fit well inside one).
#
#   bash benchmarks/hash/dither/native.sh [ARM ...]      (on a B200 or an H200, up to 90 minutes)
#
# Default arms: vllm-int8, and vllm-int8-triton on a B200 (served-row only when named: served_h1's eager engine, whose
# receipts are a side computation, not a served number). Every arm samples without FlashInfer's JIT sampler, as
# served_h1 does. Each arm runs on its leased GPU's NUMA node (numabind.py, DITHER_NUMA=0 unbound; numa.ARM.json and the
# arm's "numa" say which). Env: NATIVE_ARGS (extra native.py flags), LEASE (gpu-lease flags, default "--max-min 14"),
# VERITY_BENCH_VENV.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
[ -x "$PY" ] || PY=$(. benchmarks/harness/env.sh && echo "$VENV/bin/python") || exit 2
mkdir -p "$OUT"
# the venv's bin holds ninja, which any JIT build vLLM or FlashInfer starts needs
export PATH=$(dirname "$PY"):$PATH VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=$PWD/integrations/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH} VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HOME=${HF_HOME:-/workspace/hf}
export VLLM_CACHE_ROOT=$(mktemp -d) TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)
if [ $# -gt 0 ]; then arms=("$@"); else
  arms=(vllm-int8)
  nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -q B200 && arms+=(vllm-int8-triton)
fi
read -r -a lease <<< "${LEASE:---max-min 14}"
rc=0
for arm in "${arms[@]}"; do
  env_arm=()
  [ "$arm" = vllm-int8-triton ] && env_arm=(VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel)
  GPU_LEASE_WHO=dither/${RESEARCH_RUN_ID:-local} gpu-lease 1 --wait "${lease[@]}" -- env "${env_arm[@]}" \
    "$PY" "$HERE/numabind.py" "$OUT/numa.$arm.json" bash -c '
    q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
    nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.'"$arm"'.before.csv"
    "$0" "$1" --arm "$2" --out "$OUT" ${NATIVE_ARGS:-} > "$OUT/native-$2.log" 2>&1
    r=$?
    nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.'"$arm"'.after.csv"
    exit $r' "$PY" "$HERE/native.py" "$arm"
  r=$?
  echo "arm $arm rc $r"
  grep -h "Selected .* for \|Int8 not supported\|No compiled cutlass" "$OUT/native-$arm.log" | sort -u | head -5
  tail -2 "$OUT/native-$arm.log"
  [ $r -ne 0 ] && [ $r -ne 3 ] && rc=$r
done
rm -rf "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
exit $rc
