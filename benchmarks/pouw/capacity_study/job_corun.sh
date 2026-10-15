#!/usr/bin/env bash
# Measurement A (PoUW capacity study): an FP8 GEMM co-runner beside the Pearl-C call on one leased RTX PRO 6000.
# The ship builds off the lease (build.sh: nvcc 13.0, SASS gate, fixture); corun.py checks it on the card, then times.
# Env passed through: CORUN_SHAPES, CORUN_TORCH, CORUN_SHIP_GRIDS, CORUN_FAST_ACCUM.
set -euo pipefail
PY=${PY:-/workspace/jobs/venv312/bin/python}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
nice -n 19 bash benchmarks/pouw/pearl_c_sm120/build.sh "$d/ship" > "$OUT/build.log" 2>&1
cp benchmarks/pouw/capacity_study/corun.py "$d/ship/"
cp "$d/ship/toolchain.txt" "$d/ship/source.txt" "$d/ship/sass.json" "$d/ship/MANIFEST.sha256" "$OUT/"
git rev-parse HEAD > "$OUT/commit.txt"
gpu-lease 1 --wait --max-min 25 -- bash -c "
  set -euo pipefail
  cd '$d/ship'
  nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit,power.draw,temperature.gpu \
    --format=csv -i \"\$GPU_LEASE_UUID\" > '$OUT/gpu.txt' 2>&1 || true
  PEARLC_EXPECT_UUID=\$GPU_LEASE_UUID '$PY' corun.py check/check.json > '$OUT/corun.jsonl' 2> '$OUT/corun.stderr'
"
echo "done: $OUT/corun.jsonl"
