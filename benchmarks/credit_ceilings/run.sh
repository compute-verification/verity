#!/usr/bin/env bash
# Credit ceilings on sm_120 (CMP-12, worker W10; README.md). Run from a clean commit under `research run --cwd clone`:
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-12 --source <tree> --cwd clone --custody-ttl 12h \
#     --env GPU_LEASE_WHO=<id> -- bash benchmarks/credit_ceilings/run.sh [all|small|probe|alt|big|fused|wide|mix|lp|lp2]
#
# Builds outside the lease, then under `gpu-lease 1 --wait --max-min 20` runs credit_ceilings with nvidia-smi sampling the leased GPU's SM clock every 200 ms. Clocks are only read. Outputs go to $OUT:
# results.jsonl, the SASS of each binary, smi-*.txt/csv, and summary.md from summarize.py.
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
WHAT=${1:-all}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v)
echo "{\"run_id\":\"${RESEARCH_RUN_ID:-$(basename "${RESEARCH_RUN_DIR:-unknown}")}\",\"host\":\"$(hostname)\",\"what\":\"$WHAT\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"nvcc\":\"$($NVCC --version | tail -2 | head -1)\",\"commit\":\"$(git -C "$SRC" rev-parse HEAD 2>/dev/null)\"}" >"$OUT/meta.json"
for b in credit_ceilings; do
  if "$NVCC" "${FLAGS[@]}" -o "$WORK/$b" "$SRC/$b.cu" >"$OUT/build-$b.log" 2>&1; then
    "$(dirname "$NVCC")/cuobjdump" -sass "$WORK/$b" >"$OUT/$b.sass" 2>&1 || true
  else
    echo "build of $b failed" >>"$OUT/sections.log"
    tail -5 "$OUT/build-$b.log" >&2
  fi
done
cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
U=\$GPU_LEASE_UUID
echo "{\"gpu_uuid\":\"\$U\",\"lease_start_utc\":\"\$(date -u +%FT%TZ)\"}" >"$OUT/lease.json"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$U" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
if [ -x "$WORK/credit_ceilings" ]; then
  timeout 900 "$WORK/credit_ceilings" $WHAT >>"$OUT/results.jsonl" 2>>"$OUT/credit_ceilings.err"
  echo "credit_ceilings rc=\$?" >>"$OUT/sections.log"
fi
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$U" >"$OUT/smi-after.txt" 2>&1
EOF
gpu-lease 1 --wait --max-min 20 -- bash "$WORK/gpu.sh"
echo "gpu lease rc=$?" >>"$OUT/sections.log"
python3 "$SRC/summarize.py" "$OUT" >"$OUT/summary.md" 2>"$OUT/summarize.err" || echo "summarize failed" >>"$OUT/sections.log"
cat "$OUT/sections.log" "$OUT/summary.md" >&2
