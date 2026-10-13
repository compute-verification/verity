#!/bin/sh
# Every 5 min: copy every fixture origin/main's fixtures/artifacts.json registers to the public export that nodes and pods fetch
# fixtures from (`research data export`, the store README's export), so a newly registered fixture reaches it with nobody running
# the export by hand. The export skips what the bucket already holds, so a pass with nothing new only HEADs. Code and registry
# both come from a checkout of origin/main kept at /data/infra/fixtures-export-repo, fetched with the read-only deploy key.
# r2.env's key reads the private store and writes the public bucket; it goes only into the environment.
# Log: /data/infra/fixtures-export.log. The relay's `fixtures-public` monitor alerts when a registered fixture stays missing.
LOG=/data/infra/fixtures-export.log
REPO=/data/infra/fixtures-export-repo
exec 9> /root/fixtures-export.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/fixtures-export.lock" >> "$LOG"; exit 0; }
export GIT_SSH_COMMAND="ssh -i /data/state/secrets/verity-deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/data/state/secrets/known_hosts"
export GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8
while :; do
  (
    [ -d "$REPO/.git" ] || git clone -q --filter=blob:none --no-checkout git@github.com:danielreuter/verity.git "$REPO" || exit 1
    git -C "$REPO" fetch -q origin main && git -C "$REPO" checkout -q -f --detach FETCH_HEAD || exit 1
    set -a; . /root/.config/verity/r2.env; set +a
    cd "$REPO/tools/research" && PYTHONPATH=src RESEARCH_EXPORT_BUCKET=verity-public timeout 3000 \
      python3.12 -m research data export --store /data/infra/fixtures-export-store --repo "$REPO" --jobs 16
  ) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) pass rc=$? $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" >> "$LOG"
  sleep 300
done
