#!/usr/bin/env bash
# Share one machine's GPUs between concurrent runs: one lock file per GPU, held with flock(1).
#
#   gpu-lease N [--wait] [--on LIST] -- CMD...   take N free GPUs and run CMD with CUDA_VISIBLE_DEVICES naming them
#   gpu-lease status                             one line per GPU: index, free or held, and the holder: minutes held, pid, run
#                                                id, who, UUID, start and command
#
# A GPU is held while any process has its lock ($GPU_LEASE_DIR/<index>.lock) open: CMD and every child it starts inherit the
# descriptors, and the kernel drops the lock when the last of them exits, so a killed run never leaks a GPU and there is nothing
# to clean up.  CUDA_DEVICE_ORDER=PCI_BUS_ID makes the indices nvidia-smi's.  Too few free GPUs exits 75 (EX_TEMPFAIL) at once,
# holding nothing; --wait retries every $GPU_LEASE_POLL_S seconds (default 15) instead.
# Holding nothing is also how a CPU-only run shares the machine: it simply doesn't call gpu-lease.
# GPU_LEASE_COUNT overrides the GPU count nvidia-smi reports (tests; a machine without nvidia-smi).
# On a machine whose other GPUs belong to Kubernetes (the Nebius cluster), $GPU_LEASE_ALLOWED_FILE (default /etc/vy/direct-gpus)
# names the only indices gpu-lease may hand out, comma-separated, or `none` (every GPU is Kubernetes'): it then refuses. The file
# is written by the cluster's bring-up (cluster_up.sh), never by hand. Absent, every GPU is gpu-lease's.
#
# --on LIST takes the N GPUs only from LIST, comma-separated indices or UUIDs (GPU-...), in that order; a GPU outside the allowed
# set is refused.  CMD gets GPU_LEASE_UUID, the UUIDs it holds (comma-separated), to record with its results.  The holder is named
# by $GPU_LEASE_WHO (else $RESEARCH_LANE, else the user), so `status` says who holds what and for how long.
# GPU_LEASE_UUID_TABLE overrides the UUIDs nvidia-smi reports, comma-separated in index order (tests).
#
# Waiting is first come, first served, so a whole-node request (`gpu-lease 8 --wait`) is not starved by shorter leases starting
# around it: a --wait request that can't be met at once queues as $GPU_LEASE_DIR/wait.<ns>.<pid>, locked while it waits, and
# every later request (waiting or not) leaves the GPUs to the older waiters: a request without --wait exits 75 naming them.
# `status` lists the waiters.  A waiter that dies drops its lock, and the next reader removes its file.

set -u
dir=${GPU_LEASE_DIR:-/run/gpu-lease}

allowed_file=${GPU_LEASE_ALLOWED_FILE:-/etc/vy/direct-gpus}

gpu_indices() {
  # the indices gpu-lease may use, one per line: the allowed file's list, else 0..count-1
  local total i a
  total=$(gpu_count)
  if [ -r "$allowed_file" ]; then
    a=$(tr -d ' \n' < "$allowed_file")
    [ "$a" = none ] && return 0
    echo "$a" | tr ',' '\n' | while read -r i; do [ -n "$i" ] && [ "$i" -lt "$total" ] && echo "$i"; done
  else
    for ((i = 0; i < total; i++)); do echo "$i"; done
  fi
}

gpu_count() {
  if [ -n "${GPU_LEASE_COUNT:-}" ]; then echo "$GPU_LEASE_COUNT"; else nvidia-smi -L 2>/dev/null | grep -c '^GPU '; fi
}

uuid_table() {
  # "index uuid" per line
  if [ -n "${GPU_LEASE_UUID_TABLE:-}" ]; then
    echo "$GPU_LEASE_UUID_TABLE" | tr ',' '\n' | awk 'NF {print NR - 1, $1}'
  else
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | tr ',' ' '
  fi
}

usage() { echo "usage: gpu-lease N [--wait] [--on INDEX|UUID,...] -- CMD... | gpu-lease status" >&2; exit 2; }

live_waiters() {
  # the queued --wait requests, oldest first; a file whose waiter is gone (its lock free) is removed
  local f fd
  for f in "$dir"/wait.*; do
    [ -e "$f" ] || continue
    exec {fd}<"$f" 2>/dev/null || continue
    if flock -n -s "$fd"; then exec {fd}>&-; rm -f "$f"; continue; fi
    exec {fd}>&-
    echo "$f"
  done
}

ahead() {
  # the waiters this request must leave the GPUs to: all of them, or for a queued waiter the older ones
  live_waiters | awk -v me="${MYWAIT:-}" 'me == "" || $0 < me'
}

minutes_since() {
  local since
  since=$(echo "$1" | sed -n 's/.*since=\([^ ]*\).*/\1/p')
  since=$(date -u -d "$since" +%s 2>/dev/null) && echo "$((($(date -u +%s) - since) / 60))m "
}

status() {
  local i fd state owner since age now f
  now=$(date -u +%s)
  [ -r "$allowed_file" ] && echo "# direct-run GPUs from $allowed_file: $(tr -d '\n' < "$allowed_file") (the rest are Kubernetes')"
  for i in $(gpu_indices); do
    owner=$(cat "$dir/$i.owner" 2>/dev/null)
    exec {fd}>>"$dir/$i.lock" || { echo "$i unreadable"; continue; }
    if flock -n -s "$fd"; then state=free; owner=""; else state=held; fi
    exec {fd}>&-
    age=""
    [ -n "$owner" ] && age=$(minutes_since "$owner")
    echo "$i $state${owner:+ $age$owner}"
  done
  for f in $(live_waiters); do
    owner=$(cat "$f" 2>/dev/null)
    echo "waiting $(minutes_since "$owner")$owner"
  done
}

take() {
  # take N from CANDIDATES: TAKEN = the indices, their lock descriptors left open; 1 (holding nothing) when fewer than N are free
  local want=$1 i fd
  TAKEN=() FDS=()
  for i in "${CANDIDATES[@]}"; do
    [ "${#TAKEN[@]}" -lt "$want" ] || break
    exec {fd}>>"$dir/$i.lock" || continue
    if flock -n -x "$fd"; then TAKEN+=("$i") FDS+=("$fd"); else exec {fd}>&-; fi
  done
  if [ "${#TAKEN[@]}" -lt "$want" ]; then
    for fd in "${FDS[@]}"; do exec {fd}>&-; done
    TAKEN=() FDS=()
    return 1
  fi
}

[ $# -ge 1 ] || usage
if [ "$1" = status ]; then status; exit 0; fi
n=$1; shift
case $n in ''|*[!0-9]*|0) usage ;; esac
wait=0 on=""
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case $1 in
    --wait) wait=1 ;;
    --on) [ $# -ge 2 ] || usage; on=$2; shift ;;
    *) usage ;;
  esac
  shift
done
[ "${1:-}" = -- ] || usage
shift
[ $# -ge 1 ] || usage
mapfile -t CANDIDATES < <(gpu_indices)
total=${#CANDIDATES[@]}
if [ "$total" -eq 0 ] && [ -r "$allowed_file" ]; then
  echo "gpu-lease: no GPU is for direct runs here ($allowed_file: none): submit through the cluster's queues (sky)" >&2; exit 2
fi
if [ -n "$on" ]; then
  table=$(uuid_table)
  picked=()
  for g in $(echo "$on" | tr ',' ' '); do
    case $g in
      GPU-*) i=$(echo "$table" | awk -v u="$g" '$2 == u {print $1}') ;;
      *[!0-9]*) usage ;;
      *) i=$g ;;
    esac
    if [ -z "$i" ] || ! printf '%s\n' "${CANDIDATES[@]}" | grep -qx "$i"; then
      echo "gpu-lease: GPU $g is not one gpu-lease hands out here (${CANDIDATES[*]:-none})" >&2; exit 2
    fi
    picked+=("$i")
  done
  CANDIDATES=("${picked[@]}")
  total=${#CANDIDATES[@]}
fi
if [ "$n" -gt "$total" ]; then echo "gpu-lease: $n GPUs asked, $total are gpu-lease's here${on:+ in --on $on}" >&2; exit 2; fi
mkdir -p "$dir" 2>/dev/null
[ -w "$dir" ] || { echo "gpu-lease: $dir is not writable" >&2; exit 2; }

who=${GPU_LEASE_WHO:-${RESEARCH_LANE:-$(id -un)}}
MYWAIT=""
until [ -z "$(ahead)" ] && take "$n"; do
  if [ "$wait" != 1 ]; then
    first=$(ahead | head -1)
    if [ -n "$first" ]; then
      echo "gpu-lease: a request is waiting first ($(cat "$first" 2>/dev/null)): add --wait to queue behind it" >&2
    else
      echo "gpu-lease: fewer than $n of $total GPUs free ($(status | grep -v '^#' | tr '\n' ';'))" >&2
    fi
    exit 75
  fi
  if [ -z "$MYWAIT" ]; then
    exec {wfd}>"$dir/.wait.$$"
    flock -x "$wfd"
    echo "pid=$$ n=$n${on:+ on=$on} run=${RESEARCH_RUN_ID:--} who=${who// /_} since=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&"$wfd"
    MYWAIT="$dir/wait.$(date -u +%s%N).$$"
    mv "$dir/.wait.$$" "$MYWAIT"
  fi
  sleep "${GPU_LEASE_POLL_S:-15}"
done
if [ -n "$MYWAIT" ]; then rm -f "$MYWAIT"; exec {wfd}>&-; fi
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
[ -n "${table:-}" ] || table=$(uuid_table)
uuids=()
for i in "${TAKEN[@]}"; do
  u=$(echo "$table" | awk -v i="$i" '$1 == i {print $2}')
  uuids+=("${u:--}")
  echo "pid=$$ run=${RESEARCH_RUN_ID:--} who=${who// /_} uuid=${u:--} since=$started cmd=$(echo "$*" | cut -c1-80)" > "$dir/$i.owner"
done
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${TAKEN[*]}")
GPU_LEASE_UUID=$(IFS=,; echo "${uuids[*]}")
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID GPU_LEASE_UUID
echo "gpu-lease: GPUs $CUDA_VISIBLE_DEVICES ($GPU_LEASE_UUID) for $who, pid $$${RESEARCH_RUN_ID:+ (run $RESEARCH_RUN_ID)}" >&2
exec "$@"
