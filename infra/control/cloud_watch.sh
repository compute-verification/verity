#!/bin/sh
# The cloud watch (INF-73): our cloud infra's health and spend, from the control pod; anomalies and a 15:00Z digest in
# #agent-alerts. Code: /data/infra/cloud_watch.py and /data/infra/cloud_watch.toml (copies of infra/control/'s; standard library
# and psql). The credentials go only into the environment: r2.env and relay.env (SLACK_BOT_TOKEN, GH_TOKEN) are sourced, and
# store-index.env's URL (it has `&`) is read with sed. State: /data/cloud-watch/state.json. Log: /data/cloud-watch/watch.log
# With arguments it runs them once in the foreground, without the lock: `cloud_watch.sh once --dry-run --digest`.
DIR=/data/cloud-watch
LOG=$DIR/watch.log
CONF=/root/.config/verity
mkdir -p "$DIR"
val() { sed -n "s/^$1=//p" "$2" | tr -d "\"'"; }
watch() {
  set -a; . $CONF/r2.env; . $CONF/relay.env; set +a
  STORE_INDEXER_DATABASE_URL=$(val STORE_INDEXER_DATABASE_URL $CONF/store-index.env)
  export STORE_INDEXER_DATABASE_URL LC_ALL=C.UTF-8
  exec python3.12 /data/infra/cloud_watch.py "$@" --config /data/infra/cloud_watch.toml
}
if [ $# -gt 0 ]; then (watch "$@"); exit $?; fi
exec 9> /root/cloud-watch.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/cloud-watch.lock" >> "$LOG"; exit 0; }
while :; do
  (watch loop) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) watch exited rc=$?" >> "$LOG"
  sleep 60
done
