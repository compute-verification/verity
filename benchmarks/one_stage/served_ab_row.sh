#!/usr/bin/env bash
# A booked row of served_ab.sh on one card, bounded on the node itself: it holds for its start, runs served_ab.sh, and has
# the run off the card by a deadline the node's own clock enforces, whether or not the machine that launched it is still
# there.
#   hold      until NOT_BEFORE, then until the card is free: no process left that carries AFTER in its environment (the
#             run that had the card before, as RESEARCH_RUN_DIR=<its run dir>, which `research run` gives a run's
#             workload and not its runner), no compute process on CARD, at most FREE_MIB MiB used on it (16: an idle
#             card reads a few) and its gpu-lease free. The builds wait too: they run on the cores the run before is
#             timed on. At START_BY nothing is started any more (exit 20).
#   run       served_ab.sh in this tree, with this script's arguments and environment (ARMS, LOADS, REPS, WARM,
#             NOT_AFTER, CARD, CORES, STAGE, SERVE_ROOT, ...), once its interpreter is seen to import verity and the
#             endpoint from this tree (`served_ab.sh imports`; else exit 8, nothing started). Every process it starts
#             carries a tag in its environment.
#   deadline  at DEADLINE the tagged processes get SIGTERM, and those still there GRACE seconds later (15) get SIGKILL,
#             wherever they went (another session, a parent that died). A SIGTERM or SIGINT to this script (`research
#             run`'s timeout, a cancel) ends them at once with SIGKILL. Exit 21 when the deadline ended the run, 143 when
#             the signal did, 22 when a process outlived its SIGKILL, else served_ab.sh's exit. A live arm's period
#             that isn't closed within GRACE stays open (served_ab.sh's header says how to close it).
#   end       on every path past the hold: the run's serve dirs removed (SERVE_ROOT is memory, under /dev/shm); in
#             $OUT/row.txt the card's memory, its compute processes, its lease and how many of the run's processes are
#             left, and the card's line once more if its memory came back later than that; `nvidia-smi -q` for every
#             card in $OUT/nvidia-smi-q-end.txt (-start.txt is from just before the run).
# NOT_BEFORE, START_BY and DEADLINE are UTC times in `date -d` syntax, read against the node's clock; START_BY and
# DEADLINE are required. ROW_NOW names a command that prints the time in epoch seconds in the clock's place (tests);
# ROW_POLL is the seconds between two looks at the run (1) and ROW_HOLD_POLL between two at the card while holding (5).
#   research run --on NODE --project verity --source TREE --cwd clone --timeout T -- \
#     env CARD=1 CORES=12-19 ARMS='stock pearl deferred' NOT_BEFORE=... START_BY=... NOT_AFTER=... DEADLINE=... \
#     AFTER=RESEARCH_RUN_DIR=DIR taskset -c 12-19 bash benchmarks/one_stage/served_ab_row.sh
# `research run --timeout` counts from the run's start on the node and is the second bound: set to end after DEADLINE
# plus GRACE, it leaves the end to this script, which writes the card's last line, and it still ends the run when this
# script has died.
set -u
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
export OUT CARD=${CARD:-1} SERVE_ROOT=${SERVE_ROOT:-/dev/shm/compute-accounting-served-ab}   # served_ab.sh's own default
FREE_MIB=${FREE_MIB:-16} GRACE=${GRACE:-15} poll=${ROW_POLL:-1} hold_poll=${ROW_HOLD_POLL:-5}
LEASE=$TREE/tools/research/src/research/pods/sh/gpu_lease.sh
command -v gpu-lease > /dev/null && LEASE=$(command -v gpu-lease)
entry=SERVED_AB_ROW=$$-$(date +%s%N) child='' told='' fired='' rc=0 mib='' apps=0 lease=''

say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/row.txt"; }
now() { ${ROW_NOW:-date -u +%s}; }
at() { date -u -d "$1" +%s; }
carrying() {  # the pids of the processes whose environment has the entry $1 (NAME=value)
  local f
  for f in $(grep -lsaF -- "$1" /proc/[0-9]*/environ 2> /dev/null); do
    tr '\0' '\n' < "$f" 2> /dev/null | grep -qxF -- "$1" && { f=${f#/proc/}; echo "${f%/environ}"; }
  done
}
look() {  # the card now: mib, apps (the compute processes on it) and lease (free or held)
  local uuid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$CARD" 2> /dev/null | head -1)
  mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD" 2> /dev/null | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2> /dev/null | grep -c -F -- "${uuid:-no card}")
  lease=$(bash "$LEASE" status 2> /dev/null | awk -v c="$CARD" '$1 == c { print $2; exit }')
}
idle() { [[ $mib =~ ^[0-9]+$ ]] && [ "$mib" -le "$FREE_MIB" ] && [ "$apps" = 0 ]; }
card() { echo "card $CARD ${mib:-unread} MiB, $apps compute process(es), lease ${lease:-unread}"; }

hold() {  # 0 once the row may start; 20 at START_BY; 143 when told to stop first
  local by why='' was=''
  by=$(at "$START_BY")
  if [ -n "${NOT_BEFORE:-}" ]; then
    while [ "$(now)" -lt "$(at "$NOT_BEFORE")" ]; do [ -z "$told" ] || return 143; sleep "$poll"; done
  fi
  while :; do
    [ -z "$told" ] || return 143
    [ "$(now)" -lt "$by" ] || { say "not started: START_BY $START_BY has come${why:+ with $why}"; return 20; }
    if [ -n "${AFTER:-}" ] && [ -n "$(carrying "$AFTER")" ]; then
      why="a process still carrying $AFTER"
    else
      look
      idle && [ "$lease" = free ] && return 0
      why=$(card)
    fi
    [ "$why" = "$was" ] || say "holding: $why"
    was=$why
    sleep "$hold_poll"
  done
}

served() {  # 0 when a lifetime's interpreter imports verity and the endpoint from this tree
  local got at
  got=$(bash "$TREE/benchmarks/one_stage/served_ab.sh" imports 2> /dev/null | tail -2)
  [ -n "$got" ] || { say "not started: the venv imports neither verity nor the endpoint"; return 8; }
  for at in $got; do
    case $(realpath -m "$at") in "$(realpath "$TREE")"/*) ;;
      *) say "not started: the venv imports $at, not this tree's"; return 8 ;; esac
  done
}

end_run() {  # the run's processes ended: SIGTERM, then SIGKILL to those still there $1 seconds later (0: SIGKILL at once)
  local pids kill_at
  pids=$(carrying "$entry")
  if [ -n "$pids" ] && [ "$1" -gt 0 ]; then
    kill_at=$(($(now) + $1))
    say "SIGTERM to the run's $(wc -w <<< "$pids") process(es)"
    # shellcheck disable=SC2086
    kill -TERM $pids 2> /dev/null
    while [ -z "$told" ] && [ -n "$(carrying "$entry")" ] && [ "$(now)" -lt "$kill_at" ]; do sleep "$poll"; done
    pids=$(carrying "$entry")
  fi
  [ -z "$pids" ] || say "SIGKILL to the run's $(wc -w <<< "$pids") process(es)"
  for _ in $(seq 1 50); do   # again while any is left: one that was starting another as the signal came
    [ -n "$pids" ] || break
    # shellcheck disable=SC2086
    kill -KILL $pids 2> /dev/null
    sleep 0.2
    pids=$(carrying "$entry")
  done
}

for v in START_BY DEADLINE; do
  [ -n "${!v:-}" ] || { echo "served_ab_row.sh: $v is required, a UTC time" >&2; exit 2; }
done
for v in NOT_BEFORE START_BY DEADLINE; do
  [ -z "${!v:-}" ] || at "${!v}" > /dev/null \
    || { echo "served_ab_row.sh: $v=${!v} is not a time \`date -d\` reads" >&2; exit 2; }
done
trap 'told=1' TERM INT
trap '[ -z "$child" ] || end_run 0' EXIT
hold || exit $?
served || exit $?
nvidia-smi -q > "$OUT/nvidia-smi-q-start.txt" 2>&1
say "start: $(card); deadline $DEADLINE, then $GRACE s to SIGKILL"
env "$entry" bash "$TREE/benchmarks/one_stage/served_ab.sh" "$@" &
child=$!
deadline=$(at "$DEADLINE")
while [ -z "$told" ] && kill -0 "$child" 2> /dev/null; do
  [ "$(now)" -lt "$deadline" ] || { fired=1; say "DEADLINE $DEADLINE has come with the run still up"; break; }
  sleep "$poll"
done
[ -z "$told" ] || say "told to stop"
end_run "$([ -n "$told" ] && echo 0 || echo "$GRACE")"
left=$(carrying "$entry" | wc -l)
rc=unread
[ "$left" != 0 ] || { wait "$child" 2> /dev/null; rc=$?; }
rm -rf "${SERVE_ROOT:?}/runs/${RESEARCH_RUN_ID:-local}"
look
say "end: $(card); $left process(es) of the run left; served_ab.sh exit $rc"
if ! idle; then   # a killed process's memory comes back a little after it: the card's line once more when it has
  for _ in $(seq 1 20); do sleep "$poll"; look; idle && break; done
  say "later: $(card)"
fi
nvidia-smi -q > "$OUT/nvidia-smi-q-end.txt" 2>&1
child=''
[ "$left" = 0 ] || exit 22
[ -z "$told" ] || exit 143
[ -z "$fired" ] || exit 21
exit "$rc"
