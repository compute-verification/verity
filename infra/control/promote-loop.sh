#!/bin/sh
# `research queue promote` every minute (infra/control/README.md), so main follows a passed after-landing run within minutes;
# it skips while drive is stopped (/data/drive/stopped). Installed at /data/infra/promote-loop.sh; post_start.sh starts it.
while :; do
  [ -e /data/drive/stopped ] || timeout 600 /data/infra/drive.sh queue promote >> /data/drive/promote.log 2>&1
  sleep 60
done
