#!/bin/sh
# Every 6 h: the store's GC (`research data gc`, the store README §7.4) over the bucket, citing the notes repository the steward keeps,
# the code checkout fixtures_export.sh fetches and Linear (relay.env's app credentials; only those two variables are read). Without /data/store-gc/apply a pass only plans (its report says what would go);
# with it, a pass marks and deletes. Code: /data/infra/store-gc-src, a tools/research tree whose COMMIT file names it. A citing
# tree last fetched (.git/FETCH_HEAD) more than STALE_S ago skips the pass: a citation it lacks could lose a blob.
# Log: /data/infra/store-gc.log
LOG=/data/infra/store-gc.log
D=/data/store-gc
NOTES=/data/steward/research-notes
CODE=/data/infra/fixtures-export-repo
STALE_S=21600
exec 9> /root/store-gc.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/store-gc.lock" >> "$LOG"; exit 0; }
mkdir -p $D/reports
while :; do
  (
    now=$(date +%s)
    for t in $NOTES $CODE; do
      c=$(stat -c %Y $t/.git/FETCH_HEAD 2>/dev/null || echo 0)
      [ $((now - c)) -le $STALE_S ] || { echo "$(date -u +%FT%TZ) $t last fetched $(( (now - c) / 60 )) min ago: pass skipped"; exit 3; }
    done
    set -a; . /root/.config/verity/r2.env; set +a
    for v in LINEAR_CLIENT_ID LINEAR_CLIENT_SECRET; do
      x=$(sed -n "s/^\(export \)\{0,1\}$v=//p" /root/.config/verity/relay.env | tr -d "\"'"); [ -n "$x" ] && export "$v=$x"
    done
    APPLY=; [ -f $D/apply ] && APPLY="--apply --state $D/marks.json"
    cd /data/infra/store-gc-src && PYTHONPATH=src LC_ALL=C.UTF-8 timeout 18000 \
      python3.12 -m research data gc --store /root/store-gc-store --cite $NOTES --cite $CODE --cite-linear --cache $D/cache --work /root \
      --report $D/reports/$(date -u +%Y%m%dT%H%MZ).json $APPLY
  ) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) pass rc=$? $(cat /data/infra/store-gc-src/COMMIT 2>/dev/null)" >> "$LOG"
  sleep 21600
done
