#!/bin/sh
# The agent watch (INF-69): free cloud agents whose run is held, from the control pod. Code: /data/infra/agent_watch.py (infra/ops's
# verity_ops/agent_watch.py, standard library only). The key goes only into the environment: /data/agent-watch/cursor.env (root,
# 0600, one CURSOR_API_KEY= line) is sourced. Until it exists the loop waits and says so once. Findings: /data/agent-watch/events.jsonl,
# which infra's relay posts (infra/control/agent_watch_relay.py). Log: /data/agent-watch/watch.log
DIR=/data/agent-watch
LOG=$DIR/watch.log
mkdir -p "$DIR"
exec 9> /root/agent-watch.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/agent-watch.lock" >> "$LOG"; exit 0; }
said=
while :; do
  if [ -f "$DIR/cursor.env" ]; then
    (set -a; . "$DIR/cursor.env"; set +a; exec python3.12 /data/infra/agent_watch.py loop --log "$DIR/events.jsonl") >> "$LOG" 2>&1
    echo "$(date -u +%FT%TZ) watch exited rc=$?" >> "$LOG"
  elif [ -z "$said" ]; then
    echo "$(date -u +%FT%TZ) no $DIR/cursor.env yet: waiting" >> "$LOG"; said=1
  fi
  sleep 60
done
