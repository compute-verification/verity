#!/bin/bash
# lean_publish.sh SHA: publish the Lean build cache at SHA from VQ_LEAN_NODE (default vy-nebius-cpu-4), the machine that holds
# the cache's write key and lean's publisher, vy-lean-cache-daily, which builds and publishes its bare repository's `main`. The
# loop's steps after a landing run it (loop.sh sets RESEARCH_LEAN_PUBLISH to this file) in a checkout of the loop's clone when
# what landed touches a Lean package (merge.lean_publish), and wait only for its exit. It fast-forwards that bare repository's
# `main` to SHA as the merge mirror's push does (merge.push_mirror: research's ssh to the machine by its registered name, with
# the loop's own key, no credential), then starts the unit without waiting: a start while one runs is a no-op, and the landing
# it missed is covered by the next landing's or the daily run. Its lines go to the loop's live/lean-publish.log.
set -u
[ $# = 1 ] || { echo "usage: lean_publish.sh SHA" >&2; exit 2; }
node=${VQ_LEAN_NODE:-vy-nebius-cpu-4} py=${VQ_PYTHON:-python3.12} log=$HOME/live/lean-publish.log
export PYTHONPATH=${PYTHONPATH:-$HOME/src/tools/research/src}

publish() {
  echo "$(date -u +%FT%TZ) publish $1 on $node"
  # a full sha: git's push over ext:: takes no abbreviated one
  sha=$(git rev-parse --verify -q "$1^{commit}") || { echo "$1: no such commit in $PWD"; return 1; }
  "$py" -c 'import sys; from pathlib import Path; from research.merge import push_mirror
sys.exit(0 if push_mirror(Path.cwd(), sys.argv[2], sys.argv[1]) else 1)' "$node" "$sha" || return 1
  state=$(timeout 120 "$py" -m research pods ssh "$node" -- \
    'sudo -n systemctl start --no-block vy-lean-cache-daily.service; systemctl is-active vy-lean-cache-daily.service')
  echo "start: $state"
  case $state in active|activating) ;; *) return 1 ;; esac
}

publish "$1" >> "$log" 2>&1
rc=$?
tail -n 3 "$log"
exit $rc
