#!/usr/bin/env bash
# R4's probe on one RTX PRO 6000 of vy-nebius-2 (sm_120a), well under a minute of GPU time under one lease:
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone \
#     --send r4_probe.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/r4_probe.sh"'
# Builds outside the lease; the GPU part runs inside it; everything goes to $OUT.
set -uo pipefail
SRC=$PWD/benchmarks/gpu_model_r4
OUT=${OUT:?run under research run}
NVCC=/usr/local/cuda-13.0/bin/nvcc
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
if ! "$NVCC" -O3 -std=c++17 -arch=sm_120a "$SRC/r4_tc_red.cu" -o "$WORK/r4_tc_red" >"$OUT/build.log" 2>&1; then
  echo '{"probe":"r4_tc_red","build":"failed"}' | tee -a "$OUT/summary.jsonl"
  cat "$OUT/build.log"
  exit 1
fi
/usr/local/cuda-13.0/bin/cuobjdump -sass "$WORK/r4_tc_red" >"$OUT/sass.txt" 2>&1 || true
cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
nvidia-smi --id="\$GPU_LEASE_UUID" --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,power.max_limit --format=csv >"$OUT/smi.csv" 2>&1
nvidia-smi --id="\$GPU_LEASE_UUID" --query-gpu=clocks.sm,power.draw,clocks_event_reasons.active --format=csv -lms 100 >"$OUT/clocks.csv" 2>&1 &
SMI=\$!
"$WORK/r4_tc_red" >>"$OUT/summary.jsonl" 2>>"$OUT/stderr.log" || echo "{\"probe\":\"r4_tc_red\",\"run\":\"failed\"}" >>"$OUT/summary.jsonl"
kill \$SMI 2>/dev/null || true
EOF
gpu-lease 1 --wait --max-min 15 -- bash "$WORK/gpu.sh"
echo "lease exit $?" >>"$OUT/lease.log"
cat "$OUT/summary.jsonl"
