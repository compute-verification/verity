#!/usr/bin/env bash
# wait_run.sh RUN: wait for a run on this machine (a sibling of $RESEARCH_RUN_DIR) to end, so a job can be launched before the run
# it consumes ends; exit 0 when that run ended done with rc 0, else 9 (its state and rc printed).
set -uo pipefail
R=$(dirname "$RESEARCH_RUN_DIR")/${1:?RUN}
[ -d "$R" ] || { echo "no run $1 on $(hostname) ($R)"; exit 9; }
n=0
while :; do
  s=$(python3 -c 'import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: d = {}
t = (d.get("transitions") or [{}])[-1]
print(d.get("state") or t.get("state") or "?", t.get("rc"))' "$R/status.json")
  case ${s%% *} in running|\?) ;; *) break ;; esac
  [ $((n % 30)) != 0 ] || echo "$(date -u +%H:%MZ) waiting for $1 ($s)"
  sleep 60; n=$((n + 1))
done
echo "$1 ended: $s"
[ "$s" = "done 0" ] || exit 9
