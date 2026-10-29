#!/usr/bin/env bash
# The schedule search on one RTX PRO 6000 (sm_120): the schedule search over the forms that set JI3n_0..2, then the fit.
#
#   research run --on MACHINE --project verity --source /workspace --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=tensor -- bash benchmarks/pouw/tensor_kinds/search.sh
#
# One binary a (form, R), built in parallel outside the lease with its SASS dumped; the timed work runs under
# `gpu-lease 1 --wait --max-min 15`, nvidia-smi sampling the leased GPU's clock and power every 200 ms. Clocks are read,
# never set.
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

TAGS=()
for form in ImmaS8k16 ImmaS8m8k16 ImmaS8k32; do
  for r in 32 64; do
    tag=${form}_R$r
    TAGS+=("$tag")
    "$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -DFORM="$form" -DRR="$r" \
      -o "$WORK/$tag" "$SRC/search.cu" >"$OUT/build-$tag.log" 2>&1 &
  done
done
wait
for tag in "${TAGS[@]}"; do
  if [ ! -x "$WORK/$tag" ]; then
    tail -40 "$OUT/build-$tag.log" >&2
    exit 2
  fi
  "$CUOBJDUMP" -sass "$WORK/$tag" >"$OUT/sass/search-$tag.sass" 2>&1
done

cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
U=\$GPU_LEASE_UUID
echo "{\"gpu_uuid\":\"\$U\",\"lease_start_utc\":\"\$(date -u +%FT%TZ)\"}" >"$OUT/lease.json"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$U" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
for tag in ${TAGS[*]}; do
  timeout 300 "$WORK/\$tag" >"$OUT/search-\$tag.jsonl" 2>"$OUT/search-\$tag.err"
  echo "\$tag rc=\$?" >>"$OUT/gpu.log"
done
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-after.txt" 2>&1
EOF
gpu-lease 1 --wait --max-min 15 -- bash "$WORK/gpu.sh"
echo "gpu lease rc=$?" >>"$OUT/gpu.log"

python3 "$SRC/tensor_kinds.py" search "$OUT" >"$OUT/search.log" 2>&1
rc=$?
cat "$OUT/gpu.log" "$OUT/search.log" >&2
exit $rc
