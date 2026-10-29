#!/usr/bin/env bash
# Every native MMA form alone and beside IADD3, then the fit. TK_ARCH picks the target: 120 (one RTX
# PRO 6000, the default), 89 (one L40S; its forms build as four groups in parallel and run back to back) or 100 (one
# B200: the legacy mma.sync forms, premise (M′); group 0, int8, must build, and group 1 is dropped if it doesn't).
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source /workspace --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=tensor -- bash benchmarks/pouw/tensor_kinds/run.sh
#   research run --on vy-dc-l40s --project verity --campaign cmp-13 --source /workspace --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=ada --env TK_ARCH=89 -- bash benchmarks/pouw/tensor_kinds/run.sh
#   research run --on vy-dc-b200 --project verity --campaign eps1 --cwd source --mem-gb 8 \
#     --env GPU_LEASE_WHO=LANE --env TK_ARCH=100 -- bash benchmarks/pouw/tensor_kinds/run.sh
#
# The binaries are built and their SASS dumped outside the lease; the timed GPU work runs under `gpu-lease 1 --wait
# --max-min 15`, with nvidia-smi sampling the leased GPU's clock and power every 200 ms. Clocks are read, never set.
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
CUOBJDUMP=$(dirname "$NVCC")/cuobjdump
ARCH=${TK_ARCH:-120}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"
echo "{\"run_id\":\"${RESEARCH_RUN_ID:-unknown}\",\"host\":\"$(hostname)\",\"arch\":$ARCH,\"start_utc\":\"$(date -u +%FT%TZ)\",\"nvcc\":\"$($NVCC --version | tail -2 | head -1)\"}" >"$OUT/meta.json"

case $ARCH in
  120) GENCODE=arch=compute_120a,code=sm_120a GROUPS_=(-1) ;;
  89) GENCODE=arch=compute_89,code=sm_89 GROUPS_=(0 1 2 3) ;;
  100) GENCODE=arch=compute_100a,code=sm_100a GROUPS_=(0 1) ;;
  *) echo "TK_ARCH=$ARCH: 120, 89 or 100" >&2; exit 2 ;;
esac
pids=()
for g in "${GROUPS_[@]}"; do
  "$NVCC" -O3 -std=c++17 -gencode "$GENCODE" -DTK_SM="$ARCH" -DTK_GROUP="$g" -Xptxas -v -o "$WORK/tensor_kinds_g$g" \
    "$SRC/tensor_kinds.cu" >"$OUT/build_g$g.log" 2>&1 &
  pids+=($!)
done
built=()
for i in "${!pids[@]}"; do
  g=${GROUPS_[$i]}
  if wait "${pids[$i]}"; then
    built+=("$g")
  elif [ "$ARCH" = 100 ] && [ "$g" != 0 ]; then
    echo "group $g did not build (build_g$g.log); dropped" >>"$OUT/gpu.log"
  else
    tail -40 "$OUT/build_g$g.log" >&2
    exit 2
  fi
done
GROUPS_=("${built[@]}")
for g in "${GROUPS_[@]}"; do "$CUOBJDUMP" -sass "$WORK/tensor_kinds_g$g"; done >"$OUT/sass/tensor_kinds.sass" 2>&1

cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
U=\$GPU_LEASE_UUID
echo "{\"gpu_uuid\":\"\$U\",\"lease_start_utc\":\"\$(date -u +%FT%TZ)\"}" >"$OUT/lease.json"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$U" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
for g in ${GROUPS_[*]}; do
  timeout 900 "$WORK/tensor_kinds_g\$g" >>"$OUT/tensor_kinds.jsonl" 2>>"$OUT/tensor_kinds.err"
  echo "tensor_kinds group \$g rc=\$?" >>"$OUT/gpu.log"
done
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-after.txt" 2>&1
EOF
gpu-lease 1 --wait --max-min 25 -- bash "$WORK/gpu.sh"
echo "gpu lease rc=$?" >>"$OUT/gpu.log"

python3 "$SRC/tensor_kinds.py" "$OUT" >"$OUT/fit.log" 2>&1
rc=$?
cat "$OUT/gpu.log" "$OUT/fit.log" >&2
exit $rc
