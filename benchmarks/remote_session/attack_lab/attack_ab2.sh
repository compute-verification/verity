#!/usr/bin/env bash
# Attacks A2 and B on one RTX PRO 6000 of vy-nebius-2, cwd = the source tree (research run --cwd clone):
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone \
#     --send attack_ab2.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/attack_ab2.sh"'
# Also records, read-only and outside the lease, the host's caches (B), the GPUs' PCIe topology (C) and the clock
# state (E). Everything to $OUT.
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$PWD/benchmarks/remote_session/attack_lab
NVCC=/usr/local/cuda-13.0/bin/nvcc
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
lscpu > "$OUT/lscpu.txt" 2>&1
lscpu -C > "$OUT/lscpu-caches.txt" 2>&1
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1
nvidia-smi -q -d CLOCK,PERFORMANCE > "$OUT/clocks-q.txt" 2>&1
python3 "$SRC/codec_key.py" key "$WORK/key.bin" || { echo "key failed"; exit 3; }
nice -n 19 "$NVCC" -O3 -std=c++17 -arch=sm_120a -DKV_FIELD=1024 -DKV_MASK=1 \
  "$SRC/attack_ab2.cu" -o "$WORK/attack_ab2" -lcuda > "$OUT/build.log" 2>&1 \
  || { tail -40 "$OUT/build.log"; exit 4; }
cat > "$WORK/gpu.sh" <<EOF
set -uo pipefail
nvidia-smi --id="\$GPU_LEASE_UUID" --query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.mem,power.limit \
  --format=csv > "$OUT/smi.csv" 2>&1
"$WORK/attack_ab2" "$WORK/key.bin" >> "$OUT/summary.jsonl" 2>> "$OUT/stderr.log"
echo "probe exit \$?" >> "$OUT/lease.log"
EOF
gpu-lease 1 --wait --max-min 25 -- bash "$WORK/gpu.sh"
echo "lease exit $?" >> "$OUT/lease.log"
cat "$OUT/summary.jsonl" "$OUT/lease.log"
