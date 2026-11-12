#!/bin/sh
# Every 5 min: copy to the public export that nodes and pods fetch fixtures from (`research data export`, the store README's
# export) every fixture origin/main's fixtures/artifacts.json registers, then each fixture that a pull request marked ready
# registers at its head and main doesn't (fixtures_export.py: only those ids, and nothing of the head is run), so a newly
# registered fixture is public before a machine that reads only the export checks it, with nobody running the export by
# hand. The export never reads the bucket: each object is a conditional PUT (412 = already there), and the export store's
# record of what landed keeps a pass with nothing new from sending anything. The code that runs is origin/main's, from the
# clone at /data/infra/fixtures-export-repo, fetched over https with verity-drive's App token (gh_app_token.sh).
# Every checkout names a ref that only this script writes, under refs/export/: main's in the clone's own working tree, which
# agent_watch.sh and store_gc.sh read, and a head's in a worktree of its own. So no other fetch into the clone changes what
# a pass reads. r2.env's key reads the private store and writes the public bucket; it goes only into the environment.
#   fixtures_export.sh                    the loop, one copy, which post_start.sh starts
#   fixtures_export.sh once [--dry-run]   one pass in the foreground; --dry-run sends nothing and leaves main's checkout as it is
#   fixtures_export.sh pr N [--dry-run]   pull request N's new ids, marked or not: by hand, when a lane asks (the README)
#   fixtures_export.sh export ARGS...     the export itself, as every pass runs it: what fixtures_export.py calls for a head
# Log: /data/infra/fixtures-export.log. The relay's `fixtures-public` monitor alerts when a registered fixture stays missing.
LOG=/data/infra/fixtures-export.log
REPO=/data/infra/fixtures-export-repo
STORE=/data/infra/fixtures-export-store
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=url.https://github.com/compute-verification/verity.git.insteadOf \
  GIT_CONFIG_VALUE_0=git@github.com:compute-verification/verity.git GIT_CONFIG_KEY_1=credential.helper \
  GIT_CONFIG_VALUE_1='!/data/infra/gh_app_token.sh git'
export GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8

# One pass at a time over the clone and the export store, whoever starts it: the loop's, a `once` and a `pr N` wait here.
pass() (
  flock -n 8 || { echo "waiting for the pass that holds /root/fixtures-export-pass.lock"; flock 8; } || exit 1
  "$@"
) 8> /root/fixtures-export-pass.lock

# The export, with r2.env's key and main's code, of whatever registry its --repo names.
run_export() {
  (
    set -a; . /root/.config/verity/r2.env; set +a
    cd "$REPO/tools/research" && PYTHONPATH=src RESEARCH_EXPORT_BUCKET=verity-public timeout 3000 \
      python3.12 -m research data export --store "$STORE" --jobs 16 "$@"
  )
}

# origin/main's registry. With --dry-run nothing is fetched or checked out: the export reports what the checkout an earlier
# pass made would send.
main_pass() {
  if [ "$1" != --dry-run ]; then
    [ -d "$REPO/.git" ] || git clone -q --filter=blob:none --no-checkout git@github.com:compute-verification/verity.git "$REPO" || return 1
    git -C "$REPO" fetch -q origin +refs/heads/main:refs/export/main && git -C "$REPO" checkout -q -f --detach refs/export/main || return 1
  fi
  run_export --repo "$REPO" "$@"
}

# fixtures_export.py's passes, with the same key and code: `heads [--dry-run]`, the new ids of each pull request marked ready
# at its head, or `pr N [--dry-run]`, those of one. Each export it runs is this script's.
pulls() {
  (
    set -a; . /root/.config/verity/r2.env; set +a
    cd "$REPO/tools/research" && PYTHONPATH=src python3.12 /data/infra/fixtures_export.py "$@" --clone "$REPO" --store "$STORE" -- /data/infra/fixtures_export.sh export
  )
}

once() {
  main_pass "$@"; a=$?
  echo "pass rc=$a $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
  pulls heads "$@"; b=$?
  echo "heads pass rc=$b"
  [ "$a" -eq 0 ] && [ "$b" -eq 0 ]
}

case "${1:-}" in
  "") ;;
  once) shift; [ "$#" -eq 0 ] || [ "$*" = --dry-run ] || { echo "usage: $0 once [--dry-run]" >&2; exit 2; }; pass once "$@"; exit $? ;;
  pr) pass pulls "$@"; exit $? ;;
  export) shift; run_export "$@"; exit $? ;;
  *) echo "usage: $0 [once [--dry-run] | pr N [--dry-run]]" >&2; exit 2 ;;
esac

exec 9> /root/fixtures-export.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/fixtures-export.lock" >> "$LOG"; exit 0; }
while :; do
  pass main_pass >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) pass rc=$? $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" >> "$LOG"
  pass pulls heads >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) heads pass rc=$?" >> "$LOG"
  sleep 300
done
