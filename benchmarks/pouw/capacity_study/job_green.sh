#!/usr/bin/env bash
# Measurement A2 (PoUW capacity study): the Pearl-C call and an FP8 GEMM on disjoint SM partitions (green contexts).
# The ship builds off the lease (build.sh: nvcc 13.0, SASS gate, fixture); green.py checks it on the card, then times. Clocks and power are read, never set.
# Env passed through: GREEN_SHAPES, GREEN_SMS.
set -euo pipefail
PY=${PY:-/workspace/jobs/venv312/bin/python}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
nice -n 19 bash benchmarks/pouw/pearl_c_sm120/build.sh "$d/ship" > "$OUT/build.log" 2>&1
cp benchmarks/pouw/capacity_study/green.py "$d/ship/"
cp "$d/ship/toolchain.txt" "$d/ship/source.txt" "$d/ship/sass.json" "$d/ship/MANIFEST.sha256" "$OUT/"
git rev-parse HEAD > "$OUT/commit.txt"
gpu-lease 1 --wait --max-min 25 -- bash -c "
  set -euo pipefail
  cd '$d/ship'
  nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit,power.draw,temperature.gpu \
    --format=csv -i \"\$GPU_LEASE_UUID\" > '$OUT/gpu.txt' 2>&1 || true
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE -i \"\$GPU_LEASE_UUID\" > '$OUT/clocks_before.txt' 2>&1 || true
  PEARLC_EXPECT_UUID=\$GPU_LEASE_UUID python3 green.py check/check.json > '$OUT/green.jsonl' 2> '$OUT/green.stderr'
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE -i \"\$GPU_LEASE_UUID\" > '$OUT/clocks_after.txt' 2>&1 || true
"
echo "done: $OUT/green.jsonl"
