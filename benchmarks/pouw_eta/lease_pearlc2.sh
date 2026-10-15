#!/usr/bin/env bash
# job_pearlc.sh's second lease (ETA_LEASE=lease_pearlc2.sh; env as lease_pearlc.sh's): run.py's gate, then eta.py's SHA-512
# probe (s512_rows' launch shape, s512_seeds' rate), the par<c> schedules (c independent calls at once) over the batch
# curve, and each schedule held under NVML (its clock, power and energy a call).
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
ETA_MODE=sha python3 eta.py > "$OUT/eta_sha.jsonl" 2> "$OUT/eta_sha.err" || echo "eta sha rc=$?" >&2
echo "sha done $(date -u +%FT%TZ)" >&2
ETA_WS=8 ETA_SCHEDULES='serial deferred par2 par4 par8' \
  ETA_SHAPES='32x8192x8192 128x8192x8192 512x8192x8192 2048x8192x8192 8192x8192x8192' \
  python3 eta.py > "$OUT/eta_par.jsonl" 2> "$OUT/eta_par.err" || echo "eta par rc=$?" >&2
echo "par done $(date -u +%FT%TZ)" >&2
ETA_MODE=power ETA_W=4 ETA_SCHEDULES='serial deferred par4 nohash gemm gemm_ct plain' \
  ETA_SHAPES='32x8192x8192 128x8192x8192 8192x8192x8192' \
  python3 eta.py > "$OUT/eta_power.jsonl" 2> "$OUT/eta_power.err" || echo "eta power rc=$?" >&2
echo "power done $(date -u +%FT%TZ)" >&2
exit 0
