#!/usr/bin/env bash
# HC-PREFILL's CUDA-core probe runner: builds hcpre_dp4a outside the lease, then under one lease runs it once per
# configuration in CONFIGS (';'-separated "ROUNDS K BLOCK NC ITERS"), rows to $OUT/dp4a.jsonl. Run from the repository
# root (research run --cwd source).
set -uo pipefail
O=${OUT:?}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
B=$(mktemp -d /tmp/hcpre-dp4a.XXXXXX)
trap 'rm -rf "$B"' EXIT
"$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o "$B/hcpre_dp4a" benchmarks/pouw/filler/hcpre_dp4a.cu \
  -lcublasLt -lnvidia-ml -ldl > "$O/build.log" 2>&1 || { echo "build failed"; exit 4; }
git rev-parse HEAD > "$O/commit.txt"
q="clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active"
cat > "$B/inner.sh" <<EOF
set -uo pipefail
smi() { nvidia-smi --query-gpu=$q --format=csv,noheader,nounits \${GPU_LEASE_UUID:+-i \$GPU_LEASE_UUID} >> "$O/smi.csv"; }
smi
IFS=';' read -ra cs <<< "${CONFIGS:-4 3 128 8 65536}"
for c in "\${cs[@]}"; do
  "$B/hcpre_dp4a" \$c >> "$O/dp4a.jsonl" 2>> "$O/errors.txt"
  echo "\$c rc \$?" >> "$O/rc.txt"
done
smi
EOF
gpu-lease 1 --wait --max-min "${MAXMIN:-10}" -- bash "$B/inner.sh"
echo "lease rc $?" >> "$O/rc.txt"
