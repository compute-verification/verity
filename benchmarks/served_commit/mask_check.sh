#!/usr/bin/env bash
# mask_check.cu on one H200 in one short gpu-lease, the SM clock sampled beside it: builds it before the lease, then runs it.
# Measurement only. cwd: the source tree's root. LEASE_MIN (5). Output: $OUT/{mask_check.json, smi.csv, steps.txt, build.log}.
set -uo pipefail
. benchmarks/harness/env.sh
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/mask-check-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) nvcc $(command -v nvcc)"
  nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -I integrations/vllm/verity_vllm/commit/committer \
    -I benchmarks/served_commit benchmarks/served_commit/mask_check.cu -o "$BUILD_DIR/mask_check" >> "$OUT/build.log" 2>&1 \
    || { say "build failed"; tail -30 "$OUT/build.log"; exit 3; }
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
"$BUILD_DIR/mask_check" > "$OUT/mask_check.json" 2>> "$OUT/steps.txt"
rc=$?
say "mask_check rc $rc"
exit "$rc"
