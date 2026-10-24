#!/bin/sh
# The agent watch (INF-69): free cloud agents whose run is held, from the control pod. Code: /data/infra/agent_watch.py (infra/ops's
# verity_ops/agent_watch.py, standard library only). The key goes only into the environment: /data/agent-watch/cursor.env (root,
# 0600, one CURSOR_API_KEY= line) is sourced. Until it exists the loop waits and says so once. Findings: /data/agent-watch/events.jsonl,
# which infra's relay posts (infra/control/agent_watch_relay.py). Log: /data/agent-watch/watch.log. Coordinators come from the
# registry in fixtures_export.sh's checkout of origin/main (fetched every 5 min) and the inbox routes in the steward's notes clone
# (synced every 2 min); their doorbells from the relay's doorbells.json. Each sweep's status.json also goes to Neon's
# store_private.agent_watch_status as AGENT_WATCH_PG_URL, the store indexer's URL, read from store-index.env with
# store_index.sh's sed (its URL has `&`); without that file the watch publishes nothing.
# Workers (Task subagents) come from the same notes clone's recent notes and $DIR/workers.d/*.json.
# Pushes, for the silent rule, are read with `git ls-remote` and the read-only deploy key fixtures_export.sh fetches with. The nodes
# whose ended runs ring their sender are $DIR/nodes.json (a copy of agent_watch_nodes.json, read each sweep), their hosts the notes
# clone's machines.d.
# While $DIR/wake exists the watch starts a run on a coordinator ready to wake (--wake); remove the file to go back to recording.
# While $DIR/launch.env exists (root, 0600, one CURSOR_API_KEY= line: Daniel's user key, which the GitHub broker trusts) the
# watch relaunches a coordinator it can't resume (launch_coordinator.py from the registry's checkout) and a worker nobody
# resumed; the watch reads it only to relaunch and passes it only in the launch's environment. Without it, relaunches are recorded.
DIR=/data/agent-watch
LOG=$DIR/watch.log
CONF=/root/.config/verity
NOTES=/data/steward/research-notes
mkdir -p "$DIR"
exec 9> /root/agent-watch.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/agent-watch.lock" >> "$LOG"; exit 0; }
val() { sed -n "s/^$1=//p" "$2" | tr -d "\"'"; }
export GIT_SSH_COMMAND="ssh -i /data/state/secrets/verity-deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/data/state/secrets/known_hosts"
export GIT_TERMINAL_PROMPT=0
said=
while :; do
  if [ -f "$DIR/cursor.env" ]; then
    (set -a; . "$DIR/cursor.env"; set +a
     if [ -f $CONF/store-index.env ]; then
       AGENT_WATCH_PG_URL=$(val STORE_INDEXER_DATABASE_URL $CONF/store-index.env)
       export AGENT_WATCH_PG_URL
     fi
     exec python3.12 /data/infra/agent_watch.py loop --log "$DIR/events.jsonl" \
      --registry /data/infra/fixtures-export-repo/.agents/skills/using-slack/registry.json \
      --routes $NOTES/routes.d --notes $NOTES --git-remote git@github.com:compute-verification/verity.git \
      --nodes "$DIR/nodes.json" $([ -f "$DIR/wake" ] && echo --wake)) >> "$LOG" 2>&1
    echo "$(date -u +%FT%TZ) watch exited rc=$?" >> "$LOG"
  elif [ -z "$said" ]; then
    echo "$(date -u +%FT%TZ) no $DIR/cursor.env yet: waiting" >> "$LOG"; said=1
  fi
  sleep 60
done
