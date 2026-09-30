#!/usr/bin/env bash
# Share one machine's GPUs between concurrent runs: one lock file per GPU, held with flock(1).
#
#   gpu-lease N [--wait] -- CMD...   take N free GPUs and run CMD with CUDA_VISIBLE_DEVICES naming them
#   gpu-lease status                 one line per GPU: index, free or held, and the last holder's pid, run id and start
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

usage() { echo "usage: gpu-lease N [--wait] -- CMD... | gpu-lease status" >&2; exit 2; }

status() {
  local i fd state owner
  [ -r "$allowed_file" ] && echo "# direct-run GPUs from $allowed_file: $(tr -d '\n' < "$allowed_file") (the rest are Kubernetes')"
  for i in $(gpu_indices); do
    owner=$(cat "$dir/$i.owner" 2>/dev/null)
    exec {fd}>>"$dir/$i.lock" || { echo "$i unreadable"; continue; }
    if flock -n -s "$fd"; then state=free; owner=""; else state=held; fi
    exec {fd}>&-
    echo "$i $state${owner:+ $owner}"
  done
}

take() {
  # take N: TAKEN = the indices, their lock descriptors left open; 1 (holding nothing) when fewer than N are free
  local want=$1 i fd
  TAKEN=() FDS=()
  for i in $(gpu_indices); do
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
wait=0
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case $1 in --wait) wait=1 ;; *) usage ;; esac
  shift
done
[ "${1:-}" = -- ] || usage
shift
[ $# -ge 1 ] || usage
total=$(gpu_indices | grep -c .)
if [ "$total" -eq 0 ] && [ -r "$allowed_file" ]; then
  echo "gpu-lease: no GPU is for direct runs here ($allowed_file: none): submit through the cluster's queues (sky)" >&2; exit 2
fi
if [ "$n" -gt "$total" ]; then echo "gpu-lease: $n GPUs asked, $total are gpu-lease's here" >&2; exit 2; fi
mkdir -p "$dir" 2>/dev/null
[ -w "$dir" ] || { echo "gpu-lease: $dir is not writable" >&2; exit 2; }

until take "$n"; do
  if [ "$wait" != 1 ]; then
    echo "gpu-lease: fewer than $n of $total GPUs free ($(status | grep -v '^#' | tr '\n' ';'))" >&2
    exit 75
  fi
  sleep "${GPU_LEASE_POLL_S:-15}"
done
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
for i in "${TAKEN[@]}"; do
  echo "pid=$$ run=${RESEARCH_RUN_ID:--} since=$started" > "$dir/$i.owner"
done
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${TAKEN[*]}")
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "gpu-lease: GPUs $CUDA_VISIBLE_DEVICES for pid $$${RESEARCH_RUN_ID:+ (run $RESEARCH_RUN_ID)}" >&2
exec "$@"
