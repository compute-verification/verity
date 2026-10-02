#!/usr/bin/env bash
# check_slot.sh [--short] CMD...: run CMD (a merge check) in one of this server's check slots: a CPU range held by a flock and pinned with
# taskset, with the check pods' Python, so its suite and circuit-check keys match the pods' verdict packs.
#
#   research run --on vy-nebius-2 ... -- bash <this> uv run --locked --extra torch-cpu python tools/check/check.py [ARGS]
#
# The slots are $VY_CHECK_SLOTS, else the first line of $VY_CHECK_SLOTS_FILE (default /workspace/research/check-slots), else
# vy-nebius-1's: "0-31 32-63", on NUMA node 0, away from the Build benchmarks pinned on node 1.  Each server's file puts them
# on the NUMA node away from its pinned or timed benchmarks (vy-nebius-2: "128-159 160-191", since its timed windows run on
# GPU 0, which is NUMA node 0's).  Slot k's lock is $VY_CHECK_LOCKS/check-<a|b|c|d>.lock (default /workspace/research/locks), the
# same locks a hand-run `flock ... taskset` check takes.  Every slot busy: wait (VY_CHECK_WAIT=0 exits 75 instead).
# UV_PYTHON defaults to 3.14.7, the pods' Python; set it to override.  The check toolchain (tools/check/pod_setup.sh's pinned uv,
# elan and lake, cargo), where a server keeps it outside the default PATH, is named by the first line of $VY_CHECK_PATH_FILE
# (default /workspace/research/check-path) and goes first on CMD's PATH.
# --short: a lane's short check (a circuit-check rerun, one suite: minutes) that must not queue behind the trains' long ones. It
# takes one of the short slots, $VY_CHECK_SHORT_SLOTS (default: every check slot's CPUs, twice), locked as check-s1/check-s2, at
# nice 10: they share the train slots' CPUs, which ran 10-23% busy under three trains (vy-nebius-1, Sep 30 10:05Z), and the
# trains keep first claim on them.
set -euo pipefail
short=0
[ "${1:-}" = --short ] && { short=1; shift; }
[ $# -ge 1 ] || { echo "usage: check_slot.sh [--short] CMD..." >&2; exit 2; }
toolchain=$(head -1 "${VY_CHECK_PATH_FILE:-/workspace/research/check-path}" 2>/dev/null || true)
[ -z "$toolchain" ] || export PATH="$toolchain:$PATH"
slots=${VY_CHECK_SLOTS:-$(head -1 "${VY_CHECK_SLOTS_FILE:-/workspace/research/check-slots}" 2>/dev/null || true)}
slots=${slots:-0-31 32-63}
locks=${VY_CHECK_LOCKS:-/workspace/research/locks}
mkdir -p "$locks"
export UV_PYTHON=${UV_PYTHON:-3.14.7}
names=(a b c d)
nice=()
if [ "$short" = 1 ]; then
  every=$(echo $slots | tr ' ' ',')
  slots=${VY_CHECK_SHORT_SLOTS:-"$every $every"}
  names=(s1 s2 s3 s4)
  nice=(nice -n 10)
fi
while true; do
  i=0
  for cpus in $slots; do
    exec {fd}>>"$locks/check-${names[$i]}.lock"
    if flock -n "$fd"; then
      echo "check_slot: check-${names[$i]} (CPUs $cpus), UV_PYTHON=$UV_PYTHON" >&2
      exec "${nice[@]}" taskset -c "$cpus" "$@"
    fi
    exec {fd}>&-
    i=$((i + 1))
  done
  [ "${VY_CHECK_WAIT:-1}" = 1 ] || { echo "check_slot: every check slot is busy ($slots)" >&2; exit 75; }
  sleep "${VY_CHECK_POLL_S:-30}"
done
