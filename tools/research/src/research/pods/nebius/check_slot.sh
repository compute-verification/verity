#!/usr/bin/env bash
# check_slot.sh CMD...: run CMD (a merge check) in one of this server's check slots: a CPU range held by a flock and pinned with
# taskset, with the check pods' Python, so its suite and circuit-check keys match the pods' verdict packs.
#
#   research run --on vy-nebius-2 ... -- bash <this> uv run --locked --extra torch-cpu python tools/check/check.py [ARGS]
#
# The slots are $VY_CHECK_SLOTS, else the first line of $VY_CHECK_SLOTS_FILE (default /workspace/research/check-slots), else
# vy-nebius-1's: "0-31 32-63", on NUMA node 0, away from the Build benchmarks pinned on node 1.  Each server's file puts them
# on the NUMA node away from its pinned or timed benchmarks (vy-nebius-2: "128-159 160-191", since its timed windows run on
# GPU 0, which is NUMA node 0's).  Slot k's lock is $VY_CHECK_LOCKS/check-<a|b|c|d>.lock (default /workspace/research/locks), the
# same locks a hand-run `flock ... taskset` check takes.  Every slot busy: wait (VY_CHECK_WAIT=0 exits 75 instead).
# UV_PYTHON defaults to 3.14.7, the pods' Python; set it to override.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: check_slot.sh CMD..." >&2; exit 2; }
slots=${VY_CHECK_SLOTS:-$(head -1 "${VY_CHECK_SLOTS_FILE:-/workspace/research/check-slots}" 2>/dev/null || true)}
slots=${slots:-0-31 32-63}
locks=${VY_CHECK_LOCKS:-/workspace/research/locks}
mkdir -p "$locks"
export UV_PYTHON=${UV_PYTHON:-3.14.7}
names=(a b c d)
while true; do
  i=0
  for cpus in $slots; do
    exec {fd}>>"$locks/check-${names[$i]}.lock"
    if flock -n "$fd"; then
      echo "check_slot: check-${names[$i]} (CPUs $cpus), UV_PYTHON=$UV_PYTHON" >&2
      exec taskset -c "$cpus" "$@"
    fi
    exec {fd}>&-
    i=$((i + 1))
  done
  [ "${VY_CHECK_WAIT:-1}" = 1 ] || { echo "check_slot: every check slot is busy ($slots)" >&2; exit 75; }
  sleep "${VY_CHECK_POLL_S:-30}"
done
