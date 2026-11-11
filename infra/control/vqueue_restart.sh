#!/bin/bash
# ci restarts the live loop (vqueue_live.sh, deployed as /data/vqueue/live.sh): stop it between ticks, kill the copy holding
# the loop lock by its pid, found through the lock file (never by its command line), and start one again. Deployed as
# /data/vqueue/bin/vqueue_restart.sh, linked as /root/ci-restart.sh.
set -u
VQ=/data/vqueue/live
touch $VQ/stopped
flock -w 900 $VQ/vqueue.lock true || { echo "vqueue.lock held 900 s: not restarted"; rm -f $VQ/stopped; exit 1; }
holder() {
  for fd in /proc/[0-9]*/fd/*; do
    [ "$(readlink "$fd" 2>/dev/null)" = /root/vqueue-loop.lock ] && { p=${fd#/proc/}; p=${p%%/*}; [ "$(ps -o comm= -p $p)" = live.sh ] && echo $p; }
  done | head -n 1
}
old=$(holder)
echo "live.sh holding the loop lock: ${old:-none}"
[ -n "$old" ] && kill -TERM "$old"
flock -w 180 /root/vqueue-loop.lock true || { echo "loop lock still held 180 s after the kill"; exit 1; }
cd /data/vqueue && setsid nohup /data/vqueue/live.sh > /dev/null 2>&1 < /dev/null &
sleep 3
new=$(holder)
echo "${new:-}" > $VQ/loop.pid
echo "new live.sh pid ${new:-NONE}"
rm -f $VQ/stopped
echo "resumed at $(date -u +%T)Z"
