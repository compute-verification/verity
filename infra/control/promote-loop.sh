#!/bin/sh
# `research queue promote` every minute (infra/control/README.md), so main follows a passed after-landing run within minutes;
# it skips while drive is stopped (/data/drive/stopped). Installed at /data/infra/promote-loop.sh; post_start.sh starts it.
# Each minute ends in one `TIME pass rc=N` or `TIME stopped` line, which cloud_watch.toml's "Promote loop" reads.
LOG=/data/drive/promote.log
while :; do
  if [ -e /data/drive/stopped ]; then
    echo "$(date -u +%FT%TZ) stopped" >> "$LOG"
  else
    timeout 600 /data/infra/drive.sh queue promote >> "$LOG" 2>&1
    echo "$(date -u +%FT%TZ) pass rc=$?" >> "$LOG"
  fi
  sleep 60
done
