#!/bin/sh
# Every 5 min: bring Neon's `store_private` up to the bucket (`research data index --postgres`, the store README §5.1), and, once
# store-export.env names the export bucket, rebuild `store_public` from it (`--public`), with store-export.env's key pair if it
# has one, else r2.env's, which also writes that bucket (fixtures_export.sh). Code: /data/infra/store-index-src, a
# tools/research tree whose COMMIT file names it. The credentials go only into the environment: r2.env is sourced, and the
# unquoted lines of store-index.env (its URL has `&`) and store-export.env are read with sed. Log: /data/infra/store-index.log
LOG=/data/infra/store-index.log
CONF=/root/.config/verity
exec 9> /root/store-index.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/store-index.lock" >> "$LOG"; exit 0; }
val() { sed -n "s/^$1=//p" "$2" | tr -d "\"'"; }
while :; do
  (
    set -a; . $CONF/r2.env; set +a
    STORE_INDEXER_DATABASE_URL=$(val STORE_INDEXER_DATABASE_URL $CONF/store-index.env)
    export STORE_INDEXER_DATABASE_URL
    PUBLIC=
    if [ -f $CONF/store-export.env ]; then
      for v in RESEARCH_EXPORT_BUCKET RESEARCH_EXPORT_ACCESS_KEY_ID RESEARCH_EXPORT_SECRET_ACCESS_KEY RESEARCH_EXPORT_ENDPOINT; do
        x=$(val $v $CONF/store-export.env); [ -n "$x" ] && export "$v=$x"
      done
      if [ -z "$RESEARCH_EXPORT_BUCKET" ]; then
        echo "$(date -u +%FT%TZ) store-export.env lacks RESEARCH_EXPORT_BUCKET: no --public"
      elif [ -n "$RESEARCH_EXPORT_ACCESS_KEY_ID" ] && [ -z "$RESEARCH_EXPORT_SECRET_ACCESS_KEY" ]; then
        echo "$(date -u +%FT%TZ) store-export.env has RESEARCH_EXPORT_ACCESS_KEY_ID without _SECRET_ACCESS_KEY: no --public"
      else
        PUBLIC=--public
      fi
    fi
    cd /data/infra/store-index-src && PYTHONPATH=src LC_ALL=C.UTF-8 timeout 3000 \
      python3.12 -m research data index --store /root/store-index-store --postgres $PUBLIC --jobs 48
  ) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) pass rc=$? $(cat /data/infra/store-index-src/COMMIT 2>/dev/null)" >> "$LOG"
  sleep 300
done
