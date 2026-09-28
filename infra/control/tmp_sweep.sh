#!/bin/sh
# Hourly: remove top-level /tmp entries with nothing inside modified in 6 h, unless a process has one as its cwd, holds a file
# under it open, or names it in its environment (PYTHONPATH and the like). DRY=1 lists only. Log: /workspace/infra/tmp-sweep.log
LOG=/workspace/infra/tmp-sweep.log
while :; do
  used=$(for p in /proc/[0-9]*; do
           readlink "$p/cwd"; ls -l "$p/fd" 2>/dev/null | sed -n 's/.*-> //p'; tr '\0:=' '\n\n\n' < "$p/environ"
         done 2>/dev/null | grep '^/tmp/.' | cut -d/ -f1-3 | sort -u)
  freed=0
  for e in $(find /tmp -mindepth 1 -maxdepth 1 -mmin +360 ! -name 'tmux-*' ! -name '.*' ! -name 'ssh-*' ! -name 'systemd-*'); do
    printf '%s\n' "$used" | grep -qxF "$e" && continue
    [ -n "$(find "$e" -mmin -360 -print -quit 2>/dev/null)" ] && continue
    mb=$(du -sm "$e" 2>/dev/null | cut -f1)
    if [ "${DRY:-0}" = 1 ]; then echo "would remove $e (${mb} MB)"; else rm -rf -- "$e" && freed=$((freed + mb)); fi
  done
  [ "${DRY:-0}" = 1 ] && exit 0
  echo "$(date -u +%FT%TZ) removed ${freed} MB; $(df -Pk / | awk 'NR==2{print $5}') used" >> "$LOG"
  sleep 3600
done
