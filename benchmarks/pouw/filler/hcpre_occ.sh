#!/usr/bin/env bash
# HC-PREFILL's occupancy probe runner: builds hcpre_occ outside the lease, then under one lease runs it once per
# configuration in CONFIGS (';'-separated "LOG2 ROUNDS K W NC S MODE"; LOG2 is DFILL_CHUNK_LOG2), rows to
# $OUT/occ.jsonl. Run from the repository root (research run --cwd source).
set -uo pipefail
O=${OUT:?}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
B=$(mktemp -d /tmp/hcpre-occ.XXXXXX)
trap 'rm -rf "$B"' EXIT
"$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/hcpre_occ" \
  benchmarks/pouw/filler/hcpre_occ.cu -lcublasLt -lnvidia-ml -ldl > "$O/build.log" 2>&1 || { echo "build failed"; exit 4; }
git rev-parse HEAD > "$O/commit.txt"
q="clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active"
cat > "$B/inner.sh" <<EOF
set -uo pipefail
smi() { nvidia-smi --query-gpu=$q --format=csv,noheader,nounits \${GPU_LEASE_UUID:+-i \$GPU_LEASE_UUID} >> "$O/smi.csv"; }
smi
IFS=';' read -ra cs <<< "${CONFIGS:-18 4 3 4 4 512 0}"
for c in "\${cs[@]}"; do
  set -- \$c
  lg=\$1; shift
  DFILL_CHUNK_LOG2=\$lg "$B/hcpre_occ" "\$@" >> "$O/occ.jsonl" 2>> "$O/errors.txt"
  echo "\$lg \$* rc \$?" >> "$O/rc.txt"
done
smi
EOF
gpu-lease 1 --wait --max-min "${MAXMIN:-10}" -- bash "$B/inner.sh"
echo "lease rc $?" >> "$O/rc.txt"
