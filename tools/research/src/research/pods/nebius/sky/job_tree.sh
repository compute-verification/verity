#!/usr/bin/env bash
# The tree a Nebius Kueue job runs from:
#
#   SRC=$(bash "$SRC/tools/research/src/research/pods/nebius/sky/job_tree.sh" "$SRC")
#
# prints /workspace/jobs/src/<id>, one copy of the synced tree per content (<id> hashes its files' paths, sizes and mtimes, which a
# `research pods sync` of unchanged files keeps), shared by every job of that content and made once under a per-id lock.
# - The job user (uid 1000) can write it; the synced tree is research's (uid 1001).
# - A later sync of the same lane can't change the code under a running job.
# - Every job of one tree compiles its native taps from the same source paths, so torch's JIT build (ninja) finds them up to date.
#   A copy per pod changed every compile command and rebuilt every tap, ~3 min per GPU job under the bootstrap lock (Sep 30).
# The path is also written to /workspace/jobs/src/by-pod/<hostname>, which the job's run step reads back (`job_tree.sh --mine`), so
# setup and run agree even if the lane re-syncs in between.
set -euo pipefail
root=${VY_JOB_TREES:-/workspace/jobs/src}
if [ "${1:-}" = --mine ]; then
  cat "$root/by-pod/$(hostname)"
  exit
fi
src=${1:?usage: job_tree.sh SYNCED_TREE | --mine}
id=$(cd "$src" && find . -path ./out -prune -o -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum | cut -c1-16)
dst=$root/$id
mkdir -p "$root/by-pod"
if [ ! -e "$dst/.copied" ]; then
  exec 9> "$root/.$id.lock"
  flock 9
  if [ ! -e "$dst/.copied" ]; then
    rm -rf "$dst" "$dst.partial.$$"
    mkdir -p "$dst.partial.$$"
    cp -a "$src"/. "$dst.partial.$$"/
    mv "$dst.partial.$$" "$dst"
    touch "$dst/.copied"
  fi
  exec 9>&-
fi
echo "$dst" > "$root/by-pod/$(hostname)"
echo "$dst"
