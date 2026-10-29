#!/usr/bin/env bash
# alt_roads_egress.cu on one H200: built before the lease, run under `gpu-lease 1`, with the PCIe link and clocks
# recorded beside it. Measurement only. cwd: the source tree's root.
#   CUDA_HOME (/usr/local/cuda-13.0), LEASE_MIN (10)
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/alt-egress-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  CUDA=${CUDA_HOME:-/usr/local/cuda-13.0}
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a benchmarks/pouw/theory/alt_roads_egress.cu \
    -lcublas -o "$BUILD_DIR/egress" > "$OUT/build.log" 2>&1
  rc=$?; say "build rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-10}" -- bash "$0" leased
  exit $?
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.sm,clocks.max.sm,power.limit \
  --format=csv | tee "$OUT/gpu.csv"
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,pcie.link.gen.current,memory.used --format=csv -lms 200 \
  > "$OUT/sampler.csv" 2>&1 &
smi=$!
"$BUILD_DIR/egress" > "$OUT/egress.json" 2> "$OUT/egress.err"
rc=$?
kill "$smi" 2>/dev/null
say "egress rc $rc"
cat "$OUT/egress.json"
exit $rc
