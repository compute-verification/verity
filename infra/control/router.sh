#!/bin/sh
# comms' router (`research msg router`, INF-123): one process that reads every #chat-* channel and rings the lanes that own each
# post, restarted 30 s after it exits. Code: /data/router/src, a tools/research tree whose COMMIT file names it. Credentials:
# /data/state/secrets/router.env (root, 0600: SLACK_BOT_TOKEN, and ROUTER_ANTHROPIC_API_KEY once Daniel adds it), sourced into
# its environment only. Until both exist the loop waits and says so once. State: $RESEARCH_ROUTER_STATE (/data/router); the router
# touches $RESEARCH_ROUTER_HEARTBEAT every pass, and the control disk relay alerts in #agent-alerts when it is 5 min old. Extra
# arguments: the words of /data/router/args, if present. The registry is main's, from fixtures_export.sh's checkout (fetched every
# 5 min); the inbox routes are the steward's notes clone's. Deployed as /data/infra/router.sh. Log: /data/router/router.log
DIR=/data/router
LOG=$DIR/router.log
ENV=/data/state/secrets/router.env
mkdir -p -m 700 "$DIR"
exec 9> /root/router.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/router.lock" >> "$LOG"; exit 0; }
said=
while :; do
  if [ -f "$DIR/src/COMMIT" ] && [ -s "$ENV" ]; then
    said=
    echo "$(date -u +%FT%TZ) start $(cat "$DIR/src/COMMIT")" >> "$LOG"
    (set -a; . "$ENV"; set +a
     export RESEARCH_SLACK_REGISTRY=/data/infra/fixtures-export-repo/.agents/skills/using-slack/registry.json \
       RESEARCH_NOTES=/data/steward/research-notes RESEARCH_ROUTER_STATE=$DIR RESEARCH_ROUTER_HEARTBEAT=$DIR/heartbeat \
       PYTHONPATH=src LC_ALL=C.UTF-8
     cd "$DIR/src" && exec python3.12 -m research msg router $(cat "$DIR/args" 2>/dev/null)) >> "$LOG" 2>&1
    echo "$(date -u +%FT%TZ) router exited rc=$?" >> "$LOG"
    sleep 30
  else
    [ -z "$said" ] && echo "$(date -u +%FT%TZ) waiting for $DIR/src/COMMIT and $ENV" >> "$LOG" && said=1
    sleep 60
  fi
done
