#!/usr/bin/env bash
# job_pearlc.sh's co-residency lease (ETA_LEASE=lease_cores.sh): does hiding the tile leaves behind the next call's GEMM work
# where the format allows it today, once the GEMM leaves room on the SM?  The 128 x 128 mainloop holds the whole register file,
# so the deferred schedule there only queues the hash behind it; at the forced 64 x 64 tile (8192x8192x8192@64) one 128-thread
# hash CTA fits beside each GEMM CTA.  The kernels are lease_fast.sh's, already gated; `same` at 8192^3 rechecks the build.
# Then the served call's schedules timed at the 64 x 64 tile, and power holds at both tiles: what real overlap does to the clock.
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID PEARLC_EXPECT_UUID="$GPU_LEASE_UUID" PEARLC_SHA=fast
echo "lease $GPU_LEASE_UUID index $CUDA_VISIBLE_DEVICES $(date -u +%FT%TZ)" >&2
cd "$SHIP"
ETA_MODE=same ETA_SHAPES='8192x8192x8192' python3 eta.py > "$OUT/eta_same.jsonl" 2> "$OUT/eta_same.err"
rc=$?
if [ "$rc" -ne 0 ] || ! grep -q '"identical": true' "$OUT/eta_same.jsonl" || grep -q '"identical": false' "$OUT/eta_same.jsonl"; then
  echo "same rc=$rc or a differing buffer: nothing timed" >&2; tail -5 "$OUT/eta_same.err" >&2; exit 5
fi
echo "gate passed $(date -u +%FT%TZ)" >&2
ETA_WS=3 ETA_SHAPES='8192x8192x8192@64' \
  ETA_SCHEDULES='serial deferred deferred-a main-a main-a-nf nohash hashonly hashall gemm gemm_ct' \
  python3 eta.py > "$OUT/eta_d64.jsonl" 2> "$OUT/eta_d64.err" || echo "eta d64 rc=$?" >&2
echo "time done $(date -u +%FT%TZ)" >&2
ETA_MODE=power ETA_W=3 ETA_SHAPES='8192x8192x8192@64' ETA_SCHEDULES='serial deferred deferred-a nohash hashonly gemm gemm_ct' \
  python3 eta.py > "$OUT/eta_power_d64.jsonl" 2> "$OUT/eta_power_d64.err" || echo "eta power d64 rc=$?" >&2
ETA_MODE=power ETA_W=3 ETA_SHAPES='8192x8192x8192' ETA_SCHEDULES='deferred-a deferred-g160 main-a-nf-ct' \
  python3 eta.py > "$OUT/eta_power.jsonl" 2> "$OUT/eta_power.err" || echo "eta power rc=$?" >&2
echo "power done $(date -u +%FT%TZ)" >&2
exit 0
