#!/usr/bin/env bash
# launch.sh NAME SOURCE_DIR MEM_GB TIMEOUT_S CPUS WORKLOAD: one `research run --on auto --cwd source` in tmux session NAME,
# its launcher's output in /tmp/cir269/runs/NAME.log
set -u
name=$1 src=$2 mem=$3 timeout=$4 cpus=$5 work=$6
mkdir -p /tmp/cir269/runs
log=/tmp/cir269/runs/$name.log
printf '%s\n' "$work" > /tmp/cir269/runs/$name.cmd
cpu_flag=()
[ "$cpus" != "-" ] && cpu_flag=(--cpus "$cpus")
T="tmux -f /exec-daemon/tmux.portal.conf"
$T new-session -d -s "$name" -c /agent/repos/verity -- bash -lc \
  "research run --on auto --project verity --source $src --cwd source --mem-gb $mem --timeout $timeout ${cpu_flag[*]} \
   --wait --wait-max 1200 -- bash -lc \"\$(cat /tmp/cir269/runs/$name.cmd)\" > $log 2>&1; echo EXIT \$? >> $log"
echo "$name launched: src=$src mem=$mem timeout=$timeout cpus=$cpus"
