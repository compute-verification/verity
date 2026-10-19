#!/bin/sh
# Drive (INF-129, ci's CI-78): `research queue drive` ticks, ci's landing automation, from the control pod. Each pass fetches
# origin/main into /data/drive/verity and runs one tick from that tree's tools/research (main's code, no canary), in that
# checkout, as lane ci (RESEARCH_LANE), with its flags from tools/verity/check/drive.toml at main (`--config`, read fresh each
# tick) plus whatever /data/drive/args holds (flags that override the file's, `#` lines skipped), then sleeps 120 s.
# The queue's clone and its state are root's ~/.research/queue/ (verity, verity.state.json), on the volume.
# GitHub: the checkout's origin is https and its local credential helper reads $GH_TOKEN (relay.env's) when git
# asks, so the token is never in git's config or arguments; drive gives its own clone the same helper (`credential_args`), and
# gh (/data/drive/bin/gh) reads GH_TOKEN itself. r2.env (the store's parent key, which train.sh needs) and relay.env
# (SLACK_BOT_TOKEN for drive's posts) are sourced. The nodes are reached as `research run --on` reaches them: the notes clone's
# machines.d (RESEARCH_NOTES, the steward's clone) and ~/.runpod/ssh/runpodctl-ssh-key. The store is a cache of the bucket on the
# container disk (/root/drive-store, as the store index keeps its own): its SQLite catalog on the volume took minutes a pull.
# While /data/drive/stopped exists no tick starts (a running one finishes): `drive.sh stop` writes it and waits for the tick in
# progress to end, `drive.sh start` removes it once a dry-run push of next shows the token can push, `drive.sh status` says
# which and shows the log's tail, and `drive.sh once [FLAGS]` runs one tick here in the foreground, after the one in progress,
# with FLAGS added (`--dry-run`).
# Log: /data/drive/drive.log
DIR=/data/drive
LOG=$DIR/drive.log
REPO=$DIR/verity
CONF=/root/.config/verity
STOP=$DIR/stopped
TICK=/root/drive-tick.lock
export PATH=$DIR/bin:/root/.local/bin:$PATH GH_CONFIG_DIR=$DIR/gh-config GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8 \
  RESEARCH_LANE=ci RESEARCH_NOTES=/data/steward/research-notes RESEARCH_STORE=/root/drive-store
tick() {
  exec 8> "$TICK"; flock 8
  set -a; . $CONF/r2.env; . $CONF/relay.env; set +a
  git -C "$REPO" fetch -q origin main && git -C "$REPO" checkout -q -f --detach FETCH_HEAD || exit 1
  cd "$REPO" && PYTHONPATH=$REPO/tools/research/src exec python3.12 -m research queue drive \
    --config tools/verity/check/drive.toml $(grep -v '^#' $DIR/args 2>/dev/null) "$@"
}
case ${1:-} in
  start) (set -a; . $CONF/relay.env; set +a; git -C "$REPO" fetch -q origin +refs/heads/next:refs/remotes/origin/next \
            && git -C "$REPO" push -q --dry-run origin refs/remotes/origin/next:refs/heads/next) > /dev/null 2>&1 \
          || { echo "drive: not started: a dry-run push of next with relay.env's GH_TOKEN was refused; drive pushes next and main" \
                    "and edits and closes PRs, so the token needs Contents and Pull requests read and write on" \
                    "compute-verification/verity" >&2; exit 1; }
         rm -f "$STOP"; echo "drive: started; the loop's next pass (within 30 s) ticks"; exit 0 ;;
  stop) : > "$STOP"; echo "drive: stopped; waiting for a tick in progress to end"; flock "$TICK" true
        echo "drive: no tick running"; exit 0 ;;
  status) [ -f "$STOP" ] && echo "drive: stopped ($STOP)" || echo "drive: started"
          flock -n "$TICK" true && echo "drive: no tick running" || echo "drive: a tick is running"
          tail -n 5 "$LOG" 2>/dev/null; exit 0 ;;
  once) shift; (tick "$@"); exit $? ;;
  '') ;;
  *) echo "drive.sh [start|stop|status|once [FLAGS]]: with no argument, the loop" >&2; exit 2 ;;
esac
mkdir -p "$DIR"
exec 9> /root/drive.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/drive.lock" >> "$LOG"; exit 0; }
n=0
while :; do
  if [ -f "$STOP" ]; then
    # every 10 min, so the cloud watch reads a stopped loop as alive
    [ $((n % 20)) -ne 0 ] || echo "$(date -u +%FT%TZ) stopped: $STOP exists (drive.sh start)" >> "$LOG"
    n=$((n + 1)); sleep 30 9>&-; continue
  fi
  n=0
  (tick) >> "$LOG" 2>&1 9>&-
  echo "$(date -u +%FT%TZ) tick rc=$? $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)" >> "$LOG"
  sleep 120 9>&-
done
