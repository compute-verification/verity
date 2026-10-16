#!/usr/bin/env bash
# HC-SCHED's screens through the scorer's own driver: builds the locked dscore.cu and pscore.cu, then under one lease runs
# each of PROBES (';'-separated "dscore|pscore MODE ARGS...", as the binaries take them) and appends its row to
# $OUT/probe.jsonl; psched_probe.py summarizes the rows with the locked scoring.point_stats. A screen, not a score: no
# gates beyond each row's own receipts. Run from the repository root (research run --cwd source).
set -uo pipefail
O=${OUT:?}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
export PATH="$(dirname "$NVCC"):$PATH"
B=$(mktemp -d /tmp/psched-probe.XXXXXX)
FL=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v)
LIBS=(-lcublasLt -lnvidia-ml -ldl)
for n in dscore pscore; do
  "$NVCC" "${FL[@]}" -o "$B/$n" "benchmarks/pouw/filler/$n.cu" "${LIBS[@]}" >> "$O/build.log" 2>&1 || { echo "build $n failed"; exit 4; }
done
git rev-parse HEAD > "$O/commit.txt"
python3 -c 'import secrets; print(secrets.token_hex(16))' > "$O/coins.txt"
printf '%s\n' "${PROBES:?}" > "$O/probes.txt"
cat > "$B/inner.sh" <<EOF
set -uo pipefail
export DFILL_COINS=\$(cat "$O/coins.txt")
q="clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active"
nvidia-smi --query-gpu=\$q --format=csv,noheader,nounits \${GPU_LEASE_UUID:+-i \$GPU_LEASE_UUID} >> "$O/smi.csv"
IFS=';' read -ra P <<< "\$PROBES"
for p in "\${P[@]}"; do
  set -- \$p
  bin=\$1; shift
  "$B/\$bin" "\$@" >> "$O/probe.jsonl" 2>> "$O/errors.txt"
  echo "\$bin \$* rc \$?" >> "$O/rc.txt"
  nvidia-smi --query-gpu=\$q --format=csv,noheader,nounits \${GPU_LEASE_UUID:+-i \$GPU_LEASE_UUID} >> "$O/smi.csv"
done
EOF
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash "$B/inner.sh"
echo "lease rc $?" >> "$O/rc.txt"
rm -rf "$B"
