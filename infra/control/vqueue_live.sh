#!/bin/sh
# The verified queue, live: `research queue vqueue` (no --shadow), one tick a pass, from the pinned checkout $CODE in the
# shadow's clone $CLONE, on its own decision log under $VQ, never the shadow's. It takes the shadow's loop lock, so it never
# runs beside the shadow: stop the shadow's loop first. Its posts go to the merge thread (no MERGE_THREAD override).
# `live.sh once [FLAGS]` runs one tick in the foreground (`--dry-run` writes nothing); with no argument, the loop, which skips
# passes while $VQ/stopped exists and adds the flags in $VQ/args (`#` lines skipped).
VQ=/data/vqueue/live
FAST=/root/fast
CLONE=$FAST/vqueue-verity
CODE=$FAST/vqueue-src
CONF=/root/.config/verity
LOG=$VQ/vqueue.log
STORE=/root/vqueue-store
export PATH=/data/drive/ghw:/data/drive/bin:/root/.local/bin:$PATH GH_CONFIG_DIR=/data/drive/gh-config GIT_TERMINAL_PROMPT=0 \
  LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 RESEARCH_LANE=ci RESEARCH_NOTES=/data/steward/research-notes RESEARCH_STORE=$STORE \
  TMPDIR=/data/vqueue-tmp  # NFS: not under /root (a run's node stats the verdict pack's path: EACCES there), nor on the 40 GB root, which one candidate's three launches fill (each stages up to 8 GB of Lean bundles); NFS is slow (INF-657)
mkdir -p "$VQ" "$TMPDIR" "$STORE"
tick() {
  ( set -a; . $CONF/r2.env; . $CONF/relay.env; set +a  # relay.env: SLACK_BOT_TOKEN, which the loop's rings post with
    [ -n "${RESEARCH_AGENT_URL:-}" ] || RESEARCH_AGENT_URL=$(cat /data/drive/agent-url 2>/dev/null); export RESEARCH_AGENT_URL
    GH_TOKEN=$(/data/infra/gh_app_token.sh --cached) || exit 1; export GH_TOKEN
    cd "$CLONE" && RESEARCH_DRIVE=1 PYTHONPATH=$CODE/tools/research/src exec python3.12 -m research queue vqueue --repo "$CLONE" \
      --log $VQ/decisions.jsonl --state $VQ/state.json --lock $VQ/vqueue.lock --once "$@" )
}
case ${1:-} in
  once) shift; tick "$@"; exit $? ;;
  '') ;;
  *) echo "live.sh [once FLAGS]: with no argument, the live loop" >&2; exit 2 ;;
esac
exec 9> /root/vqueue-loop.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/vqueue-loop.lock" >> "$LOG"; exit 0; }
while :; do
  if [ -f "$VQ/stopped" ]; then sleep 30 9>&-; continue; fi
  tick $(grep -v '^#' $VQ/args 2>/dev/null) >> "$LOG" 2>&1 9>&-
  echo "$(date -u +%FT%TZ) tick rc=$? $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)" >> "$LOG"
  find "$TMPDIR" -maxdepth 1 -name 'train-*' -mmin +40 -exec rm -rf {} + 2>/dev/null  # a run beside a check reads its staging tree only until it registers
  find "$TMPDIR" -maxdepth 1 -name 'check-lean-deps-*' -mmin +40 -exec rm -rf {} + 2>/dev/null  # a launch's Lean bundles, read only until its run registers
  sleep 60 9>&-
done
