#!/usr/bin/env bash
# The scan's per-link-window answers held by verity/certifier/v0's rows form (NET-75) on a CPU machine: two warden
# rehearsals at the test class's schedule (scan_zk.sh's SC_REHEARSE), honest and void (the fake answering past the
# warden's delay, SC_LATE_MS: every connection fails late), each proved as one group of SC_GROUP link-windows, every
# unit drawn; then scan_period.py's cases (honest, late, and the honest run's tampered round) through the period's
# mirror and network-period, both from SC_PERIOD_REF (the rows form's branch) in a worktree. From a clone, in a slot
# of network-accounting's pool:
#   research run --on MACHINE --source . --project verity --as LANE --cwd clone --timeout 180m --env SC_MEM_GB=64 -- \
#     env CHECK_SLOTS=/workspace/research/locks/research/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/scan_period.sh
# env: SC_MEM_GB (scan_zk.sh's)  SC_GROUP (default 2)  SC_LATE_MS (default 800)  SC_PERIOD_REF (default
#      cursor/allp-cert-warden-form-061a)
# Kept in $OUT: honest/ and late/ (each scan_zk.sh's out/: the rehearsal's public files and the scan's kept files) and
# period/ (each case's files and period.json).
set -uo pipefail
export HZ_CONFIGS=sczk HZ_MEM_GB=${SC_MEM_GB:?}
SC_CACHE=${SC_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$SC_CACHE" 2>/dev/null && [ -w "$SC_CACHE" ]; then export HZ_CACHE=$SC_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
export FLOCK_PROVE=$BIN FLOCK_SELFTEST=$SBIN FLOCK_VERIFY=$LEAN HZ_PYTHON=$PY
export SC_CLASS=test SC_GROUP=${SC_GROUP:-2} SC_REHEARSE=1 SC_REQUESTS=${SC_REQUESTS:-4} SC_RATE=${SC_RATE:-1} \
  SC_TAIL=${SC_TAIL:-3} SC_PROMPT_MAX=${SC_PROMPT_MAX:-40} SC_MAX_TOKENS=${SC_MAX_TOKENS:-8}
REF=${SC_PERIOD_REF:-cursor/allp-cert-warden-form-061a}
log "period: network-period and the mirror from $REF"
{ git fetch -q origin "$REF" && git worktree add -q --detach "$W/period" FETCH_HEAD \
  && (cd "$W/period/verity/core/protocols/network/lean" && lake build network-period); } > "$OUT/build-period.log" 2>&1 &
built=$!
declare -A rc
for run in honest late; do
  mkdir -p "$W/$run"
  log "period: scan_zk.sh on the $run rehearsal"
  RESEARCH_RUN_DIR=$W/$run SC_FIRST_MS=$([ "$run" = late ] && echo "${SC_LATE_MS:-800}" || echo 100) \
    bash benchmarks/network_traces/scan_zk.sh > "$OUT/scan-$run.log" 2>&1
  rc[$run]=$?
  cp -r "$W/$run/out" "$OUT/$run"
  log "period: the $run run, scan_zk.sh rc ${rc[$run]}: $(grep -h '^SCZK .*answers:' "$OUT/$run/sczk.log" | cut -c1-600)"
done
wait "$built" || { tail -30 "$OUT/build-period.log"; log "period: no network-period"; exit 7; }
NP=$W/period/verity/core/protocols/network/lean/.lake/build/bin/network-period
PYTHONPATH=$PYTHONPATH:$SRC/verity/core/difftest/python "$PY" benchmarks/network_traces/scan_period.py \
  --mirror "$W/period/verity/core/protocols/network/tests/period_case.py" --lean "$NP" --out "$OUT/period" \
  "honest=$OUT/honest/warden,$OUT/honest/sczk" "late=$OUT/late/warden,$OUT/late/sczk" \
  "tampered=$OUT/honest/warden,$OUT/honest/sczk,tampered" 2>&1 | tee -a "$OUT/period.log"
p=${PIPESTATUS[0]}
log "period: scans ${rc[honest]} ${rc[late]}, period check $p"
[ "${rc[honest]}" = 0 ] && [ "${rc[late]}" = 0 ] && exit "$p"
exit 1
