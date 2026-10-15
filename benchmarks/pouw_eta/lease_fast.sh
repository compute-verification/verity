#!/usr/bin/env bash
# job_pearlc.sh's fast-SHA-512 lease (ETA_LEASE=lease_fast.sh, ETA_PROBE_VARIANTS set): the gates first, nothing timed unless
# both pass: run.py's fixture check, every buffer bit for bit, with hash_fast.cuh's kernels (PEARLC_SHA=fast, both variants)
# and with the reference kernels; then eta.py's `same`, the committed buffers of every step and of the served call's graph
# identical to the reference kernels' at seven shapes.  Then `sha2` (the kernels by launch-bound variant), the served call
# and its what-ifs timed with the reference and with the fast kernels, the fast call's batch curve, and the power holds.
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID PEARLC_EXPECT_UUID="$GPU_LEASE_UUID"
echo "lease $GPU_LEASE_UUID index $CUDA_VISIBLE_DEVICES $(date -u +%FT%TZ)" >&2
cd "$SHIP"
for sha in fast ref; do
  PEARLC_SHA=$sha VARIANTS='sm120 sm120-unpromoted' SHAPES='' python3 run.py check/check.json > "$OUT/run_py_$sha.jsonl" \
    2> "$OUT/run_py_$sha.err"
  rc=$?
  tail -3 "$OUT/run_py_$sha.err" >&2
  if [ "$rc" -ne 0 ]; then echo "run.py ($sha) rc=$rc: the fixture gate failed, nothing timed" >&2; exit "$rc"; fi
done
ETA_MODE=same python3 eta.py > "$OUT/eta_same.jsonl" 2> "$OUT/eta_same.err"
rc=$?
grep -c '"identical": true' "$OUT/eta_same.jsonl" >&2
if [ "$rc" -ne 0 ] || grep -q '"identical": false' "$OUT/eta_same.jsonl"; then
  echo "same rc=$rc or a differing buffer: the in-call gate failed, nothing timed" >&2; tail -5 "$OUT/eta_same.err" >&2; exit 5
fi
echo "gates passed $(date -u +%FT%TZ)" >&2
ETA_MODE=sha2 python3 eta.py > "$OUT/eta_sha2.jsonl" 2> "$OUT/eta_sha2.err" || echo "sha2 rc=$?" >&2
echo "sha2 done $(date -u +%FT%TZ)" >&2
S3='32x8192x8192 128x8192x8192 8192x8192x8192'
PEARLC_SHA=ref ETA_WS=3 ETA_SCHEDULES='serial deferred main-a main-a-nf nohash' ETA_SHAPES="$S3" \
  python3 eta.py > "$OUT/eta_ref.jsonl" 2> "$OUT/eta_ref.err" || echo "eta ref rc=$?" >&2
PEARLC_SHA=fast ETA_WS=3 ETA_SHAPES="$S3" ETA_SCHEDULES='serial serial+tt deferred deferred-a deferred-g140 deferred-g160 main-a
  main-a-nf main-a-nf-ct hashonly hashall nohash gemm gemm_ct plain' \
  python3 eta.py > "$OUT/eta_fast.jsonl" 2> "$OUT/eta_fast.err" || echo "eta fast rc=$?" >&2
echo "time done $(date -u +%FT%TZ)" >&2
PEARLC_SHA=fast ETA_WS=3 ETA_SCHEDULES='serial deferred' \
  ETA_SHAPES='256x8192x8192 512x8192x8192 1024x8192x8192 2048x8192x8192 4096x8192x8192' \
  python3 eta.py > "$OUT/eta_fast_curve.jsonl" 2> "$OUT/eta_fast_curve.err" || echo "eta curve rc=$?" >&2
PEARLC_SHA=fast ETA_MODE=power ETA_W=3 ETA_SHAPES='8192x8192x8192' \
  ETA_SCHEDULES='serial deferred main-a main-a-nf hashonly hashall nohash gemm gemm_ct plain' \
  python3 eta.py > "$OUT/eta_power.jsonl" 2> "$OUT/eta_power.err" || echo "eta power rc=$?" >&2
echo "power done $(date -u +%FT%TZ)" >&2
exit 0
