#!/usr/bin/env bash
# The timed window's legs under one lease, from the run tree's root (--cwd clone: the PoUS and joint legs build Lean in it):
#   GPU_LEASE_WHO=<id> LEGS=A,B,C,F [AT=<the booking's start, UTC> RELEASE='<its booking's text>'] \
#     bash benchmarks/resource_accounting/window.sh
# Each leg (`leg` below) is run.sh into $OUT/<leg> with that leg's env, and TUNED for the tuned arms:
#  1. every leg's PHASE=prepare, over one scratch, on the CPUs: the ship, the PoUS verdict and the tagger, each built once; a
#     leg whose preparation fails is dropped;
#  2. at AT (the booked window's start, UTC; at once when unset or past), `gpu-lease $LEASE --max-min $MINUTES` (default
#     `8 --wait --timed`, 60) with every host thread under `taskset -c $HOST_CPUS` (default 128-159) runs each leg's PHASE=gpu
#     LEASE=held in LEGS' order, the chain on GPU CHAIN_GPU (default 4, on HOST_CPUS's NUMA node); a leg starts only with its
#     `about` minutes left before the lease's last minute, under its cap or what is left, the lesser, as its LEASE_MIN;
#  3. RELEASE, a fixed string, deletes the one line of the booked windows file that holds it, once the lease has ended or when
#     no leg is left to run: the booking's release;
#  4. every leg's PHASE=verify, on the CPUs.
# $OUT/window.jsonl records each leg's phases (start, end, exit, LEASE_MIN) or why it was skipped; the exit is the first
# failing leg's. The rehearsal is the same run untimed on one GPU: LEASE='1 --wait' MINUTES=30 CHAIN_GPU= and the legs that fit.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run, or give OUT}/out}; mkdir -p "$OUT"
LEGS=${LEGS:-A,B,C,F}; LEASE=${LEASE:-8 --wait --timed}; MINUTES=${MINUTES:-60}; HOST_CPUS=${HOST_CPUS-128-159}
# TUNED: the tuned harness library's build where node 2 keeps it (a workload has no
# store remote to fetch it from)
CHAIN_GPU=${CHAIN_GPU-4}; TUNED=${TUNED:-/workspace/pouw/harness/build/05a44c565456c043}
WINDOWS=${GPU_LEASE_WINDOWS:-/workspace/pouw/fill/windows}
[ "$MINUTES" -le 60 ] || { echo "RUN_FAIL_LEASE: MINUTES=$MINUTES is over the booked 60" >&2; exit 2; }
stamp() { echo "ra-window: $* at $(date -u +%H:%M:%SZ)"; }
NATIVE=native-bf16,native-fp8,native-bf16-tuned,native-fp8-tuned; PEARLS=pearl-c,pearl-c-deferred,pearl-c-nohash
# a leg's minutes (about, cap) and run.sh env
leg() {
  case $1 in
    A) echo "8 10 ARMS=$NATIVE,$PEARLS" ;;
    B) echo "5 7 ARMS=$NATIVE SIDECAR=1" ;;
    C) echo "6 8 SHAPES=o,down,cube8192 LENGTHS=8,128 ARMS=$NATIVE,$PEARLS" ;;
    D) echo "9 11 JOINT=joint-v1 SHAPES=o BATCHES=1,128 ARMS=native-bf16,native-fp8 POUS_AUDITS=100 SMI_MS=10 ROUND_MIN=6" ;;
    E) echo "20 21 JOINT=separate,inproc,joint SHAPES=o BATCHES=1,128 ARMS=native-bf16,native-fp8 POUS_AUDITS=100 ROUND_MIN=5" ;;
    F) echo "7 9 POUS=1 POUS_IDLE=1 SHAPES=o BATCHES=1,128 ARMS=native-fp8,pearl-c" ;;
    *) return 1 ;;
  esac
}
record() { printf '%s\n' "$1" >> "$OUT/window.jsonl"; }
# LEG PHASE [K=V...]: run.sh's PHASE for LEG, into $OUT/LEG, its output appended to $OUT/LEG/PHASE.log
phase() {
  local l=$1 p=$2 t0 rc=0 env; shift 2
  read -r _ _ env <<< "$(leg "$l")"
  mkdir -p "$OUT/$l"; t0=$(date -u +%FT%TZ)
  # shellcheck disable=SC2086,SC2163
  ( export $env "$@" TUNED HOST_CPUS CHAIN_GPU; OUT=$OUT/$l SCRATCH=$S LEG=$l PHASE=$p bash benchmarks/resource_accounting/run.sh ) \
    >> "$OUT/$l/$p.log" 2>&1 || rc=$?
  record "{\"leg\": \"$l\", \"phase\": \"$p\", \"start\": \"$t0\", \"end\": \"$(date -u +%FT%TZ)\", \"rc\": $rc, \"env\": \"$env $*\"}"
  return "$rc"
}
release() {
  [ -n "${RELEASE:-}" ] || return 0
  local n; n=$(grep -cF -- "$RELEASE" "$WINDOWS" || true)
  if [ "$n" = 1 ]; then sed -i "$(grep -nF -- "$RELEASE" "$WINDOWS" | cut -d: -f1)d" "$WINDOWS"; stamp "released the booking in $WINDOWS"
  else stamp "RELEASE matches $n lines of $WINDOWS, not one: the booking is left as it is"; fi
}

if [ "${1:-}" = inside ]; then
  end=$(( $(date +%s) + MINUTES * 60 - 60 )); first=0
  for l in ${LEGS//,/ }; do
    read -r about cap _ <<< "$(leg "$l")"
    left=$(( (end - $(date +%s)) / 60 ))
    if [ "$left" -lt "$about" ]; then
      stamp "leg $l skipped: $left min left, it takes about $about"
      record "{\"leg\": \"$l\", \"phase\": \"gpu\", \"skipped\": \"$left min left of the lease, the leg takes about $about\"}"
      continue
    fi
    min=$(( cap < left ? cap : left ))
    stamp "leg $l under the lease, LEASE_MIN $min"
    rc=0; phase "$l" gpu LEASE=held LEASE_MIN="$min" || rc=$?
    stamp "leg $l exited $rc"
    [ "$rc" = 0 ] || [ "$first" != 0 ] || first=$rc
  done
  exit "$first"
fi

S=$(mktemp -d /workspace/tmp/ra-window.XXXXXX); trap 'rm -rf "$S"' EXIT
for l in ${LEGS//,/ }; do leg "$l" > /dev/null || { stamp "RUN_FAIL_LEGS: no leg $l (A to F)"; exit 2; }; done
git rev-parse HEAD > "$OUT/tree.txt"
stamp "legs $LEGS: preparing"
ready=()
for l in ${LEGS//,/ }; do
  if phase "$l" prepare; then ready+=("$l"); else stamp "leg $l dropped: its preparation failed ($OUT/$l/prepare.log)"; fi
done
rc=0
if [ ${#ready[@]} = 0 ]; then
  stamp "RUN_FAIL_PREPARE: no leg is ready"; release; exit 13
fi
if [ -n "${AT:-}" ]; then
  at=$(date -u -d "$AT" +%s); now=$(date +%s)
  if [ "$at" -gt "$now" ]; then stamp "waiting for $AT"; sleep $(( at - now )); fi
fi
stamp "legs ${ready[*]} under gpu-lease $LEASE --max-min $MINUTES, host threads on ${HOST_CPUS:-any CPU}, the chain on GPU ${CHAIN_GPU:-first leased}"
pin=(); [ -z "$HOST_CPUS" ] || pin=(taskset -c "$HOST_CPUS")
export OUT S MINUTES TUNED HOST_CPUS CHAIN_GPU
set +e
# shellcheck disable=SC2086
LEGS=$(IFS=,; echo "${ready[*]}") gpu-lease $LEASE --max-min "$MINUTES" -- "${pin[@]}" bash "$0" inside 2>&1 | tee "$OUT/lease.log"
rc=${PIPESTATUS[0]}
set -e
stamp "lease ended, its legs exited $rc"
release
for l in "${ready[@]}"; do
  vrc=0; phase "$l" verify || vrc=$?
  [ "$vrc" = 0 ] || [ "$rc" != 0 ] || rc=$vrc
done
stamp "done, exit $rc"
exit "$rc"
