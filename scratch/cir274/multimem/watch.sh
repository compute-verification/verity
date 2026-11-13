#!/usr/bin/env bash
# REQBLOCK: run "${@:2}" with its RSS read each second, killed by its PID past 48 GB (the lead's bound); its peak to
# $OUT/watch_$1.txt either way. Exits with the command's status.
set -u
tag=$1
shift
"$@" &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  rss=$(awk '/VmRSS/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
  [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
  if [ "${rss:-0}" -gt $((48 * 1024 * 1024)) ]; then
    echo "killed at rss_kb=$rss" >> "$OUT/watch_$tag.txt"
    kill -9 "$pid"
  fi
  sleep 1
done
wait "$pid"
rc=$?
echo "$tag rc=$rc peak_rss_kb=$peak" | tee -a "$OUT/watch_$tag.txt"
exit "$rc"
