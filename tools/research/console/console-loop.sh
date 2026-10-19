#!/bin/sh
# Verity's live-console panels, store group, every 10 minutes (verity_console.py; a dry run until
# /root/.config/verity/panels.key exists). Started at boot by /post_start.sh; a second copy exits.
cd /workspace/console || exit 1
exec 9>/workspace/console/.loop.lock
flock -n 9 || { echo "console-loop: already running" >&2; exit 0; }
export RESEARCH_STORE=/workspace/steward/store
val() { sed -n "s/^$1=//p" "$2" | tr -d "\"'"; }
while :; do
  [ "$(stat -c %s console.log 2>/dev/null || echo 0)" -gt 2000000 ] && mv -f console.log console.log.1
  ( set -a; . /root/.config/verity/r2.env; set +a
    STORE_INDEXER_DATABASE_URL=$(val STORE_INDEXER_DATABASE_URL /root/.config/verity/store-index.env)
    export STORE_INDEXER_DATABASE_URL
    timeout 1500 /usr/local/bin/python3.12 verity_console.py --group store \
      --research-src /data/infra/store-index-src/src ) >> console.log 2>&1
  sleep 600
done
