#!/usr/bin/env bash
# job_pearlc.sh's third lease (ETA_LEASE=lease_pearlc3.sh; env as lease_pearlc.sh's): run.py's gate, then the served call
# with s512_rows at 32 threads a CTA (serial-r32) beside the served serial over the batch curve, and both held under NVML at
# prefill.
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID PEARLC_EXPECT_UUID="$GPU_LEASE_UUID"
echo "lease $GPU_LEASE_UUID index $CUDA_VISIBLE_DEVICES $(date -u +%FT%TZ)" >&2
cd "$SHIP"
VARIANTS=sm120 SHAPES='32x8192x8192 128x8192x8192 8192x8192x8192' REPS=9 python3 run.py check/check.json \
  > "$OUT/run_py.jsonl" 2> "$OUT/run_py.err"
rc=$?
tail -5 "$OUT/run_py.err" >&2
if [ "$rc" -ne 0 ]; then echo "run.py rc=$rc: the gate failed, nothing timed" >&2; exit "$rc"; fi
echo "gate passed $(date -u +%FT%TZ)" >&2
ETA_WS=3 ETA_SCHEDULES='serial serial-r32 deferred' \
  ETA_SHAPES='128x8192x8192 256x8192x8192 512x8192x8192 1024x8192x8192 2048x8192x8192 4096x8192x8192 8192x8192x8192' \
  python3 eta.py > "$OUT/eta_r32.jsonl" 2> "$OUT/eta_r32.err" || echo "eta r32 rc=$?" >&2
echo "r32 done $(date -u +%FT%TZ)" >&2
ETA_MODE=power ETA_W=3 ETA_SCHEDULES='serial serial-r32' ETA_SHAPES='8192x8192x8192' \
  python3 eta.py > "$OUT/eta_power.jsonl" 2> "$OUT/eta_power.err" || echo "eta power rc=$?" >&2
echo "power done $(date -u +%FT%TZ)" >&2
exit 0
