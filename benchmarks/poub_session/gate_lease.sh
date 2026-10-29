#!/usr/bin/env bash
# A gate run's prover lease with a deadline (gate_run.sh's step 3 when LEASE_BY is set; cwd = the source tree):
# `gpu-lease N ARGS -- bash run.sh`, asked from LEASE_FROM (UTC, default now) and given up at LEASE_BY, when it exits 75
# having held nothing. run.sh starts at most once.
#
# Without --wait, gpu-lease queues to the node's agent exactly as a --wait request does (the wait file doesn't say which)
# and leaves after GPU_LEASE_AGENT_WAIT_S, which is set to what is left until LEASE_BY, so a lease the agent hasn't
# granted by then never starts. A request refused at once (a booked window it would run into, or gpu-lease's own rules
# while the agent is down) is asked again: every second for the first minute after LEASE_FROM, so it queues as soon
# as a window ends, then every 10 s. The GPUs' UUIDs are read once at launch (GPU_LEASE_UUID_TABLE), so the asks run no
# nvidia-smi, which a timed window ending at LEASE_FROM mustn't see.
#
#   LEASE_BY=2026-10-24T13:14Z LEASE_FROM=2026-10-24T12:45Z gate_lease.sh 8 --max-min 100 --mem-gb 256
set -u
n=${1:?usage: gate_lease.sh N GPU-LEASE-ARGS...}
shift
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
LEASE=${GPU_LEASE:-/workspace/pouw/infra/bin/gpu-lease}
by=$(date -u -d "${LEASE_BY:?}" +%s) || exit 2
from=$(date -u -d "${LEASE_FROM:-now}" +%s) || exit 2
if [ -z "${GPU_LEASE_UUID_TABLE:-}" ]; then
  GPU_LEASE_UUID_TABLE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' ' | paste -sd,) || exit 3
fi
export GPU_LEASE_UUID_TABLE
now=$(date -u +%s)
if [ "$from" -gt "$now" ]; then
  echo "gate_lease: asking for $n GPUs from $(date -u -d "@$from" +%FT%TZ), until $(date -u -d "@$by" +%FT%TZ)"
  sleep $((from - now))
fi
while :; do
  left=$((by - $(date -u +%s)))
  [ "$left" -gt 0 ] || { echo "gate_lease: no $n GPUs leased by $(date -u -d "@$by" +%FT%TZ): nothing ran"; exit 75; }
  # shellcheck disable=SC2016
  GPU_LEASE_AGENT_WAIT_S=$left "$LEASE" "$n" "$@" -- \
    bash -c 'date -u +%FT%TZ > "$0/leased"; exec bash benchmarks/poub_session/run.sh' "$O"
  rc=$?
  if [ -f "$O/leased" ] || [ $rc != 75 ]; then exit $rc; fi
  if [ $(($(date -u +%s) - from)) -lt 60 ]; then sleep 1; else sleep 10; fi
done
