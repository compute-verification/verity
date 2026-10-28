#!/usr/bin/env bash
# hash_cost.py on one H200 in one short gpu-lease: hm96-sha512 rows alone, the Merkle tree, and the leaves beside the served int8 GEMM
# (Llama-3.2-1B at T 8,192) at a range of hash shares, the SM clock sampled beside it all. The libraries are built before the lease.
# Measurement only. cwd: the source tree's root.
#   VERITY_BENCH_VENV (the node's torch + triton environment), CUDA_HOME (/usr/local/cuda-13.0), LEASE_MIN (20), and hash_cost.py's
#   ROWS, TOTAL_GB, THREADS, WARPS, T, HASH_MS, REPS.
set -uo pipefail
. benchmarks/harness/env.sh
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/hash-cost-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) python $VENV/bin/python into $BUILD_DIR"
  "$VENV/bin/python" benchmarks/served_commit/hash_cost.py build > "$OUT/build.log" 2>&1
  rc=$?; say "build rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-20}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 200 \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
"$VENV/bin/python" benchmarks/served_commit/hash_cost.py 2>&1 | tee "$OUT/hash_cost.log"
rc=${PIPESTATUS[0]}
say "hash_cost rc $rc"
exit "$rc"
