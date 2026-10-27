#!/usr/bin/env bash
# On one A100 (sm_80): every native mma.sync form alone and beside a scalar lane, the scalar pipes, duo
# beside IMMA and the integer forms' exactness, then the gate verdicts and the fit.
#
#   research run --on vy-dc-a100 --project verity --source /workspace --cwd source \
#     --env GPU_LEASE_WHO=ampere -- bash benchmarks/pouw/tensor_kinds/run_sm80.sh [GROUP...]
#
# The binaries (one a GROUP of sm80.cu, built in parallel) and gpu_kit's SASS timing gate run outside the lease; the
# timed GPU work runs under `gpu-lease 1 --wait --max-min 20`, with nvidia-smi sampling the GPU's clock and power every
# 200 ms. Clocks are read, never set. Every output is under $OUT.
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda-12.4}
export CUDA_HOME=$CUDA
NVCC=$CUDA/bin/nvcc
GROUPS_=${*:-0 1 2 3 4 5 6 7}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass" "$OUT/build"
echo "{\"run_id\":\"${RESEARCH_RUN_ID:-unknown}\",\"host\":\"$(hostname)\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"nvcc\":\"$($NVCC --version | tail -2 | head -1)\",\"groups\":\"$GROUPS_\"}" >"$OUT/meta.json"

pids=()
for g in $GROUPS_; do
  "$NVCC" -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_80 --no-compress -Xptxas -v -DGROUP="$g" -o "$WORK/sm80_g$g" \
    "$SRC/sm80.cu" -ldl >"$OUT/build/g$g.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=2; done
if [ $rc -ne 0 ]; then
  for g in $GROUPS_; do grep -m20 -i "error" "$OUT/build/g$g.log" >&2; done
  exit 2
fi
for g in $GROUPS_; do "$CUDA/bin/cuobjdump" -sass "$WORK/sm80_g$g" >"$OUT/sass/g$g.sass" 2>&1; done

# gpu_kit's timing gate on every timed kernel, before any is timed
python3 "$SRC/sm80.py" gate "$OUT" $(for g in $GROUPS_; do echo "$WORK/sm80_g$g"; done) >"$OUT/gate.log" 2>&1
echo "gate rc=$?" >>"$OUT/gate.log"
tail -3 "$OUT/gate.log" >&2

cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
U=\$GPU_LEASE_UUID
echo "{\"gpu_uuid\":\"\$U\",\"lease_start_utc\":\"\$(date -u +%FT%TZ)\"}" >"$OUT/lease.json"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$U" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
for g in $GROUPS_; do
  timeout 600 "$WORK/sm80_g\$g" >"$OUT/g\$g.jsonl" 2>"$OUT/g\$g.err"
  echo "g\$g rc=\$? \$(date -u +%FT%TZ)" >>"$OUT/gpu.log"
done
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-after.txt" 2>&1
EOF
gpu-lease 1 --wait --max-min 20 -- bash "$WORK/gpu.sh"
echo "gpu lease rc=$?" >>"$OUT/gpu.log"

python3 "$SRC/sm80.py" analyze "$OUT" >"$OUT/fit.log" 2>&1
rc=$?
cat "$OUT/gpu.log" >&2
tail -40 "$OUT/fit.log" >&2
exit $rc
