#!/usr/bin/env bash
# M1 of EPS1 TENSOR on one RTX PRO 6000 (sm_120): every native MMA form alone and beside IADD3, then the fit.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source /workspace --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=tensor -- bash benchmarks/pouw/tensor_kinds/run.sh
#
# The binary is built and its SASS dumped outside the lease; the timed GPU work runs under `gpu-lease 1 --wait
# --max-min 15`, with nvidia-smi sampling the leased GPU's clock and power every 200 ms. Clocks are read, never set.
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
CUOBJDUMP=$(dirname "$NVCC")/cuobjdump
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"
echo "{\"run_id\":\"${RESEARCH_RUN_ID:-unknown}\",\"host\":\"$(hostname)\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"nvcc\":\"$($NVCC --version | tail -2 | head -1)\"}" >"$OUT/meta.json"

if ! "$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$WORK/tensor_kinds" \
    "$SRC/tensor_kinds.cu" >"$OUT/build.log" 2>&1; then
  tail -40 "$OUT/build.log" >&2
  exit 2
fi
"$CUOBJDUMP" -sass "$WORK/tensor_kinds" >"$OUT/sass/tensor_kinds.sass" 2>&1

cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
U=\$GPU_LEASE_UUID
echo "{\"gpu_uuid\":\"\$U\",\"lease_start_utc\":\"\$(date -u +%FT%TZ)\"}" >"$OUT/lease.json"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$U" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
timeout 600 "$WORK/tensor_kinds" >"$OUT/tensor_kinds.jsonl" 2>"$OUT/tensor_kinds.err"
echo "tensor_kinds rc=\$?" >"$OUT/gpu.log"
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-after.txt" 2>&1
EOF
gpu-lease 1 --wait --max-min 15 -- bash "$WORK/gpu.sh"
echo "gpu lease rc=$?" >>"$OUT/gpu.log"

python3 "$SRC/tensor_kinds.py" "$OUT" >"$OUT/fit.log" 2>&1
rc=$?
cat "$OUT/gpu.log" "$OUT/fit.log" >&2
exit $rc
