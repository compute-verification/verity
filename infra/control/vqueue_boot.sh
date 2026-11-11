#!/bin/bash
# The landing loop after a boot, started by post_start.sh once git is new enough for its merges. A RunPod reset wipes the
# container disk, which holds the loop's clone ($CLONE), its code ($CODE: a worktree of the clone at the commit $VQ/pin names,
# which vqueue_deploy.sh writes, else origin's ci/vqueue-live), its store and the run-book's links in /root, so this rebuilds
# what is missing, from origin and the volume, then starts vqueue_live.sh (deployed as /data/vqueue/live.sh). The loop holds a
# lock, so one already running is left alone, and $VQ/stopped keeps it idle as it did before the boot.
set -u
VQ=${VQ:-/data/vqueue/live}
BIN=${BIN:-/data/vqueue/bin}
LIVE=${LIVE:-/data/vqueue/live.sh}
FAST=${FAST:-/root/fast}
HOME_DIR=${HOME_DIR:-/root}
URL=${URL:-https://github.com/compute-verification/verity.git}
CLONE=$FAST/vqueue-verity
CODE=$FAST/vqueue-src
export PATH=/data/drive/ghw:/data/drive/bin:/root/.local/bin:$PATH GH_CONFIG_DIR=/data/drive/gh-config GIT_TERMINAL_PROMPT=0
say() { echo "$(date -u +%FT%TZ) $*"; }
mkdir -p "$VQ" "$FAST" "$FAST/vqueue-merged" "$HOME_DIR/vqueue-store"
if [ ! -d "$CLONE/.git" ]; then
  say "cloning $URL into $CLONE"
  rm -rf "$CLONE"
  auth=()
  helper=$(git config -f "$BIN/clone.gitconfig" credential.helper 2>/dev/null) && auth=(-c credential.helper= -c "credential.helper=$helper")
  git "${auth[@]}" clone -q "$URL" "$CLONE" || { say "clone failed: the loop is not started"; exit 1; }
  # the saved config holds the push credential helper and the merge drivers the candidates' merges name
  [ -f "$BIN/clone.gitconfig" ] && cp "$BIN/clone.gitconfig" "$CLONE/.git/config"
fi
if [ ! -e "$CODE/.git" ]; then
  pin=$(cat "$VQ/pin" 2>/dev/null)
  git -C "$CLONE" fetch -q origin ci/vqueue-live || { say "fetching ci/vqueue-live failed: the loop is not started"; exit 1; }
  [ -n "$pin" ] || pin=$(git -C "$CLONE" rev-parse FETCH_HEAD)
  git -C "$CLONE" cat-file -e "$pin^{commit}" 2>/dev/null || git -C "$CLONE" fetch -q origin "$pin" \
    || { say "pin $pin is not on origin: the loop is not started"; exit 1; }
  git -C "$CLONE" worktree prune
  rm -rf "$CODE"
  git -C "$CLONE" worktree add -q --detach "$CODE" "$pin" || { say "no worktree at $pin: the loop is not started"; exit 1; }
  say "code at $pin"
fi
ln -sfn /data/infra/graph2/g2remote.sh "$HOME_DIR/g2remote.sh"
ln -sfn "$BIN/vqueue_deploy.sh" "$HOME_DIR/ci-deploy.sh"
ln -sfn "$BIN/vqueue_restart.sh" "$HOME_DIR/ci-restart.sh"
ln -sfn "$BIN/vqueue_drop.sh" "$HOME_DIR/ci-drop-candidate.sh"
[ -f "$VQ/stopped" ] && say "$VQ/stopped exists: the loop starts idle"
(cd "$(dirname "$LIVE")" && setsid -f "$LIVE" > /dev/null 2>&1 < /dev/null)
say "launched $LIVE at $(git -C "$CODE" rev-parse --short=10 HEAD) (a loop already running keeps its lock, and this copy exits)"
