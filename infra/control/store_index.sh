#!/bin/sh
# Every 5 min: bring Neon's `store_private` up to the bucket (`research data index --postgres`, the store README §5.1). Code:
# /data/infra/store-index-src, a tools/research tree whose COMMIT file names it. The credentials go only into the environment:
# r2.env is sourced, and store-index.env's one unquoted line (its URL has `&`) is read with sed. Log: /data/infra/store-index.log
LOG=/data/infra/store-index.log
exec 9> /root/store-index.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/store-index.lock" >> "$LOG"; exit 0; }
while :; do
  (
    set -a; . /root/.config/verity/r2.env; set +a
    STORE_INDEXER_DATABASE_URL=$(sed -n 's/^STORE_INDEXER_DATABASE_URL=//p' /root/.config/verity/store-index.env | tr -d "\"'")
    export STORE_INDEXER_DATABASE_URL
    cd /data/infra/store-index-src && PYTHONPATH=src LC_ALL=C.UTF-8 timeout 3000 \
      python3.12 -m research data index --store /root/store-index-store --postgres --jobs 48
  ) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) pass rc=$? $(cat /data/infra/store-index-src/COMMIT 2>/dev/null)" >> "$LOG"
  sleep 300
done
