#!/bin/bash
# Re-arm the two control daemons by PID FILE only (Daniel 03:19Z: no broad pkill for critical control daemons).
# usage: rearm.sh <cap-usd> <deadline-epoch> <pod-max-hours>
set -u
CAP=${1:?cap}; DL=${2:?deadline-epoch}; PM=${3:?pod-max-hours}
cd /root/dm
echo "$CAP" > CAP
stop(){ # $1 = pidfile ; kills exactly the recorded pid iff its cmdline is the expected daemon
  local f=$1 want=$2 pid; [ -s "$f" ] || return 0; pid=$(cat "$f")
  if [ -r /proc/$pid/cmdline ] && tr "\0" " " < /proc/$pid/cmdline | grep -q -- "$want"; then kill "$pid"; sleep 1; fi
  rm -f "$f"
}
stop budget_cap.pid budget_cap.py
stop deadline.pid deadline.sh
setsid nohup python3 /root/dm/budget_cap.py run --tag 2026-09-20-r17-podside --cap 240 --cap-file /root/dm/CAP --pod-max-hours "$PM" --rate-max 45 --own-prefix cp- --allow-multiple >> /root/dm/budget.out 2>&1 < /dev/null &
echo $! > budget_cap.pid
setsid nohup bash /root/dm/deadline.sh cp- "$DL" 3agtydjyy8mrr3 >> /root/dm/deadline.out 2>&1 < /dev/null &
echo $! > deadline.pid
sleep 2
echo "$(date -u +%FT%TZ) REARM cap=$CAP deadline=$(date -u -d @$DL +%FT%TZ) pod-max=${PM}h budget_cap.pid=$(cat budget_cap.pid) deadline.pid=$(cat deadline.pid)" >> /root/dm/dm.log
tail -1 /root/dm/dm.log
