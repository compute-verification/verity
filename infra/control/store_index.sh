#!/bin/sh
# Every 5 min: bring Neon's `store_private` up to the bucket (`research data index --postgres`, the store README §5.1), and, once
# store-export.env names the export bucket, rebuild `store_public` from it (`--public`), with store-export.env's key pair if it
# has one, else r2.env's, which also writes that bucket (fixtures_export.sh). Code: /data/infra/store-index-src, a
# tools/research tree whose COMMIT file names it. The credentials go only into the environment: r2.env is sourced, and the
# unquoted lines of store-index.env (its URL has `&`) and store-export.env are read with sed. Log: /data/infra/store-index.log
# Each pass's start, end, exit code and commit replace $PASS as it ends, and the watchers tree's loop_metrics.py sends its length
# to the metrics door in the background, under `timeout 60` and without the lock's descriptor; the next pass reaps it. A failed
# send's reason is said in the log once, with `metrics send rc=1`, while the door answers so ($SENT keeps it); one its timeout
# killed is `metrics send rc=124` each time. The loop never waits on a send.
LOG=/data/infra/store-index.log
CONF=/root/.config/verity
PASS=/data/infra/store-index-pass.json
SENT=/data/infra/store-index-send.json
WATCHERS_SRC=/data/watchers/src
exec 9> /root/store-index.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/store-index.lock" >> "$LOG"; exit 0; }
val() { sed -n "s/^$1=//p" "$2" | tr -d "\"'"; }
while :; do
  start=$(date +%s)
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
  rc=$?
  end=$(date +%s)
  commit=$(cat /data/infra/store-index-src/COMMIT 2>/dev/null)
  echo "$(date -u +%FT%TZ) pass rc=$rc $commit" >> "$LOG"
  printf '{"start": %s, "end": %s, "rc": %s, "commit": "%s"}\n' "$start" "$end" "$rc" "$commit" > "$PASS.tmp" && mv "$PASS.tmp" "$PASS"
  if [ -n "$sending" ]; then    # the last pass's send, ended within its 60 s: reaped, never waited on
    wait "$sending" || { sent=$?; echo "$(date -u +%FT%TZ) metrics send rc=$sent" >> "$LOG"; }
  fi
  PYTHONPATH=$WATCHERS_SRC/tools/research/src timeout 60 python3.12 -s "$WATCHERS_SRC/infra/control/loop_metrics.py" index \
    --pass "$PASS" --state "$SENT" >> "$LOG" 2>&1 9>&- &
  sending=$!
  sleep 300
done
