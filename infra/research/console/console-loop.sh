#!/bin/sh
# Verity's live-console panels, store group, every 10 minutes (verity_console.py; a dry run until
# /root/.config/verity/panels.key exists). Started at boot by /post_start.sh; a second copy exits.
cd /workspace/console || exit 1
exec 9>/workspace/console/.loop.lock
flock -n 9 || { echo "console-loop: already running" >&2; exit 0; }
export RESEARCH_STORE=/workspace/steward/store
while :; do
  [ "$(stat -c %s console.log 2>/dev/null || echo 0)" -gt 2000000 ] && mv -f console.log console.log.1
  ( set -a; . /root/.config/verity/r2.env; set +a
    timeout 1500 /usr/local/bin/python3.12 verity_console.py --group store ) >> console.log 2>&1
  sleep 600
done
