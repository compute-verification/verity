#!/bin/sh
# Every 5 min: copy every fixture origin/main's artifacts.json registers to the public export that nodes and pods fetch
# fixtures from (`research data export`, the store README's export), so a newly registered fixture reaches it with nobody running
# the export by hand. The export never reads the bucket: each object is a conditional PUT (412 = already there), and the
# export store's record of what landed keeps a pass with nothing new from sending anything. Code and registry
# both come from a checkout of origin/main kept at /data/infra/fixtures-export-repo, fetched over https with verity-drive's App token (gh_app_token.sh).
# r2.env's key reads the private store and writes the public bucket; it goes only into the environment.
# Log: /data/infra/fixtures-export.log. The relay's `fixtures-public` monitor alerts when a registered fixture stays missing.
LOG=/data/infra/fixtures-export.log
REPO=/data/infra/fixtures-export-repo
W=/data/infra/fixtures-export-next
DONE=/data/infra/fixtures-export-candidates
# Then each candidate's registry (the verified queue's builds, origin's queue/<sha12> branches), with main's code: a fixture a
# marked PR registers is public within a pass of its candidate's build, not only once it lands. A candidate's tree never
# changes, so one exported whole (its sha in $DONE) is not exported again.
candidates() {
  git -C "$REPO" fetch -q --prune origin '+refs/heads/queue/*:refs/remotes/origin/queue/*' || return 1
  for c in $(git -C "$REPO" for-each-ref --format='%(objectname)' refs/remotes/origin/queue/); do
    grep -qx "$c" "$DONE" 2>/dev/null && continue
    (
      git -C "$REPO" worktree remove --force "$W" 2>/dev/null; rm -rf "$W"
      git -C "$REPO" worktree add -q --detach "$W" "$c" || exit 1
      set -a; . /root/.config/verity/r2.env; set +a
      cd "$REPO/tools/research" && PYTHONPATH=src RESEARCH_EXPORT_BUCKET="$(grep -o 'RESEARCH_EXPORT_BUCKET=[a-z0-9][a-z0-9.-]*' "$0" | head -1 | cut -d= -f2)" \
        timeout 3000 python3.12 -m research data export --store /data/infra/fixtures-export-store --repo "$W" --jobs 16
      rc=$?; git -C "$REPO" worktree remove --force "$W"; exit $rc
    )
    rc=$?; [ $rc -ne 0 ] || echo "$c" >> "$DONE"
    echo "$(date -u +%FT%TZ) candidate $c pass rc=$rc"
  done
}
exec 9> /root/fixtures-export.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/fixtures-export.lock" >> "$LOG"; exit 0; }
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=url.https://github.com/compute-verification/verity.git.insteadOf \
  GIT_CONFIG_VALUE_0=git@github.com:compute-verification/verity.git GIT_CONFIG_KEY_1=credential.helper \
  GIT_CONFIG_VALUE_1='!/data/infra/gh_app_token.sh git'
export GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8
while :; do
  (
    [ -d "$REPO/.git" ] || git clone -q --filter=blob:none --no-checkout git@github.com:compute-verification/verity.git "$REPO" || exit 1
    git -C "$REPO" fetch -q origin main && git -C "$REPO" checkout -q -f --detach FETCH_HEAD || exit 1
    set -a; . /root/.config/verity/r2.env; set +a
    cd "$REPO/tools/research" && PYTHONPATH=src RESEARCH_EXPORT_BUCKET=verity-public timeout 3000 \
      python3.12 -m research data export --store /data/infra/fixtures-export-store --repo "$REPO" --jobs 16
  ) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) pass rc=$? $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" >> "$LOG"
  candidates >> "$LOG" 2>&1
  sleep 300
done
