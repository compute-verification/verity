#!/usr/bin/env bash
# M5's falsification probes on one RTX PRO 6000 of vy-nebius-2 (sm_120a), a few minutes under one lease:
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone \
#     --send m5_axioms.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/m5_axioms.sh"'
# Builds each probe on its own (one that fails to build is reported, the rest still run) and writes everything to $OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${M5_SRC:-$PWD/benchmarks/gpu_model_m5}
[ -f "$SRC/m5_reinterp.cu" ] || SRC=$HERE
OUT=${OUT:?run under research run}
NVCC=/usr/local/cuda-13.0/bin/nvcc
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
for p in m5_reinterp m5_l2red m5_dram m5_compress; do
  extra=""
  [ "$p" = m5_compress ] && extra="-lcuda"
  if "$NVCC" -O3 -std=c++17 -arch=sm_120a "$SRC/$p.cu" -o "$WORK/$p" $extra >"$OUT/build-$p.log" 2>&1; then
    /usr/local/cuda-13.0/bin/cuobjdump -sass "$WORK/$p" >"$OUT/sass-$p.txt" 2>&1 || true
  else
    echo "{\"probe\":\"$p\",\"build\":\"failed\"}" >>"$OUT/summary.jsonl"
  fi
done
# Between the two QMMAs of each reinterpretation kernel: every SASS op, so a conversion would show.
if [ -f "$OUT/sass-m5_reinterp.txt" ]; then
  awk '/Function :/{f=$3} /QMMA|IMMA|HMMA/{q[f]++} /F2FP|F2F|I2F|F2I|I2FP|F2IP/{c[f]++} END{for (k in q) printf "{\"probe\":\"reinterp-sass\",\"kernel\":\"%s\",\"mma\":%d,\"conversions\":%d}\n", k, q[k], c[k]+0}' \
    "$OUT/sass-m5_reinterp.txt" >>"$OUT/summary.jsonl"
fi
cat >"$WORK/gpu.sh" <<EOF
set -uo pipefail
nvidia-smi --id="\$GPU_LEASE_UUID" --query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit,power.max_limit,pcie.link.gen.max,pcie.link.width.max --format=csv >"$OUT/smi.csv" 2>&1
nvidia-smi --id="\$GPU_LEASE_UUID" --query-gpu=clocks.sm,power.draw,clocks_event_reasons.active --format=csv -lms 250 >"$OUT/clocks.csv" 2>&1 &
SMI=\$!
for p in m5_compress m5_reinterp m5_l2red m5_dram; do
  [ -x "$WORK/\$p" ] || continue
  "$WORK/\$p" >>"$OUT/summary.jsonl" 2>>"$OUT/stderr-\$p.log" || echo "{\"probe\":\"\$p\",\"run\":\"failed\",\"exit\":\$?}" >>"$OUT/summary.jsonl"
done
kill \$SMI 2>/dev/null || true
EOF
gpu-lease 1 --wait --max-min 15 -- bash "$WORK/gpu.sh"
echo "lease exit $?" >>"$OUT/lease.log"
cat "$OUT/summary.jsonl"
