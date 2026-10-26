#!/bin/sh
# The control disk relay (control_disk_alert.py) on the pod itself, restarted when it exits: the root disk and the steward's notes
# sync, posted in #agent-alerts. relay.env (GH_TOKEN, SLACK_BOT_TOKEN) goes only into its environment.
# Code: /data/infra/control-disk/control_disk_alert.py, run with /data/infra/control-disk/research-src (main's tools/research; COMMIT
# names it) and the registry copied beside it. Deployed as /data/infra/control-disk/run.sh. With arguments it runs once in the
# foreground, without the lock: `run.sh --once --dry-run`. Log: /data/infra/control-disk/relay.log
D=/data/infra/control-disk
LOG=$D/relay.log
run() {
  set -a; . /data/state/config-verity/relay.env; set +a
  export PYTHONPATH=$D/research-src/src CONTROL_POD_LOCAL=1 CONTROL_DISK_SEEN=$D/seen.json RESEARCH_SLACK_REGISTRY=$D/registry.json LC_ALL=C.UTF-8
  cd $D && exec /usr/local/bin/python3.12 -s "${SCRIPT:-$D/control_disk_alert.py}" "$@"
}
if [ $# -gt 0 ]; then (run "$@"); exit $?; fi
exec 9> /root/control-disk.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/control-disk.lock" >> "$LOG"; exit 0; }
while :; do
  (run) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) relay exited rc=$?" >> "$LOG"
  sleep 60
done
