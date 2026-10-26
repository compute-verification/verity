#!/bin/sh
# The watchers supervisor (watchers.py, INF-131) on the control pod, started again 10 s after it exits: every loop that watches the
# system and used to run on an agent VM, from one pushed commit's tree (/data/watchers/src; its COMMIT file names it) and the loop
# table /data/watchers/watchers.toml. Deployed as /data/watchers/run.sh; post_start.sh starts it. To restart the supervisor and
# every loop (after a deploy): kill "$(cat /data/watchers/supervisor.pid)". Log: /data/watchers/supervisor.log; each loop's
# in /data/watchers/log/; their states in /data/watchers/status.json.
D=/data/watchers
LOG=$D/supervisor.log
exec 9> /root/watchers.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/watchers.lock" >> "$LOG"; exit 0; }
while :; do
  PYTHONPATH=$D/src/tools/research/src LC_ALL=C.UTF-8 /usr/local/bin/python3.12 -s $D/src/infra/control/watchers.py run \
    --config $D/watchers.toml >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) supervisor exited rc=$?" >> "$LOG"
  sleep 10
done
