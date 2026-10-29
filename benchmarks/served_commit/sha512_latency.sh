#!/usr/bin/env bash
# sha512_latency.cu on one H200 in one short gpu-lease, the SM clock sampled beside it: builds it with the committer's
# sha512.cuh rolled and unrolled (SHA_ROLL 1 and 0) before the lease, then runs both. Measurement only. cwd: the source
# tree's root. LEASE_MIN (5). Output: $OUT/{latency_r0.json, latency_r1.json, smi.csv, steps.txt}.
set -uo pipefail
. benchmarks/harness/env.sh
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/sha512-latency-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) nvcc $(command -v nvcc)"
  for roll in 0 1; do
    nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -DSHA_ROLL=$roll \
      -I integrations/vllm/verity_vllm/commit/committer benchmarks/served_commit/sha512_latency.cu \
      -o "$BUILD_DIR/latency_r$roll" >> "$OUT/build.log" 2>&1 || { say "build r$roll failed"; tail -30 "$OUT/build.log"; exit 3; }
  done
  say "built"
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-5}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 200 \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
rc=0
for roll in 0 1; do
  "$BUILD_DIR/latency_r$roll" > "$OUT/latency_r$roll.json" 2>> "$OUT/steps.txt" || rc=$?
  say "latency_r$roll rc $rc"
done
exit "$rc"
