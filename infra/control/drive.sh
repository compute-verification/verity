#!/bin/sh
# Drive: `research queue drive` ticks, ci's landing automation, from the control pod. Each pass fetches
# origin/main into /data/drive/verity and runs one tick from that tree's tools/research, checked out at the commit
# /data/drive/deployed pins (`drive.sh deploy SHA`, a landed commit) or else at main's tip, in that
# checkout, as lane ci (RESEARCH_LANE), with its flags from tools/verity/check/drive.toml at main (read fresh each tick, their one
# source) plus whatever /data/drive/args holds (`#` lines skipped): only flags drive.toml's `command_line` lets override its keys
# (`--on`), and `--priority-file /data/drive/priority`, drive's priority list, whose order is the landing order (a `--first`
# there refuses the tick); a `--config FILE` there is refused unless every key it sets has main's value; then sleeps 120 s.
# The queue's state is root's ~/.research/queue/verity.state.json, on the volume. Both git clones, drive's checkout ($REPO) and
# the queue's (~/.research/queue/verity, a symlink), live on the container disk under $FAST: on the volume's NFS git cost
# about 4.3 min a tick. `fast_clones` seeds each from its copy on the volume ($DIR/verity,
# verity.volume) when the container is new, so a pod restart costs one copy, and the next fetch brings it up to date.
# GitHub: both clones' origin is https and their local credential helper is `gh_app_token.sh git`, which hands git the
# verity-drive App's token when git asks, so the token is never in git's config or arguments; drive gives its own clone the same
# helper (`credential_args`). gh is $DIR/ghw/gh, which sets GH_TOKEN the same way for each call before running /data/drive/bin/gh.
# The token is an installation token (1 h, contents and pull requests write on verity) from
# /data/state/secrets/verity-drive.pem, reused for 45 min (`gh_app_token.sh --cached`): a tick can outlast one token, and the
# push of next at the end of a 90-min tick failed with an expired one.
# r2.env (the store's parent key, which train.sh needs) and relay.env (SLACK_BOT_TOKEN for drive's posts) are sourced.
# Drive posts as ci's handle, which `research msg` allows only ci's holder: RESEARCH_AGENT_URL is the caller's, else /data/drive/agent-url
# (ci's coordinator URL, one line; ci rewrites it at a handover). The nodes are reached as `research run --on` reaches them: the notes clone's
# machines.d (RESEARCH_NOTES, the steward's clone) and ~/.runpod/ssh/runpodctl-ssh-key. The store is a cache of the bucket on the
# container disk (/root/drive-store, as the store index keeps its own): its SQLite catalog on the volume took minutes a pull. Its
# objects/ and attempts/ are links into $DIR/store on the volume (`store_links`): they filled the pod's 20 GiB root to 94%,
# and the store writes each file beside its destination, so a link never makes a write cross filesystems.
# While /data/drive/stopped exists no tick starts (a running one finishes): `drive.sh stop` writes it and waits for the tick in
# progress to end, `drive.sh start` removes it once a dry-run push of next shows the token can push, `drive.sh status` says
# which and shows the log's tail, and `drive.sh once [FLAGS]` runs one tick here in the foreground, after the one in progress,
# with FLAGS added (`--dry-run`). `drive.sh queue ARGS` runs `research queue ARGS` the same way, between ticks, against drive's
# state (the lander's `queue eject N --by ci --reason ...`, which edits verity.state.json; the next tick rebuilds next), but
# `queue baseline SHA --by ci` and `queue promote` (a loop's, every minute or so), which fast-forward main and write nothing of
# drive's, run beside a tick (`unlocked`), so main follows a passed after-landing run within minutes, and `drive.sh merge
# ARGS` runs `research merge ARGS` (`--push`) in the queue's clone on branch main, reset to origin's: $REPO is a blob:none
# partial clone, where git fetches each object it lacks as a pack of its own. All of it runs with RESEARCH_DRIVE=1: GitHub's rulesets let only
# the verity-drive App move main and next, and `checked` once infra adds it there (drive lands the reviewed
# prefix on checked, fast-forwards main only to a checked commit whose after-landing run passed every step on the merge
# gate, and merges a main pushed outside it into checked); `research` refuses those pushes anywhere else.
# Log: /data/drive/drive.log
DIR=/data/drive
LOG=$DIR/drive.log
FAST=/root/fast
REPO=$FAST/verity
QUEUE=/root/.research/queue/verity
CONF=/root/.config/verity
STOP=$DIR/stopped
TICK=/root/drive-tick.lock
# TMPDIR is on the volume: a train's check.py --record holds ~8 GiB in check-lean-deps-* at its peak, which took the pod's 20 GiB
# root overlay to 95%. post_train.py's lock is under it too (tempfile.gettempdir()), so change it only between ticks.
# A train's checkout (~430 MB) is on the container disk ($TRAIN_TREES): on the volume, uv and Python's imports from it left
# check.py in NFS waits for minutes, and launches registered no run. `prune_trees` removes one an hour after its last
# change once no process runs in it, so a few an hour stay a few GB.
# RESEARCH_EXECUTOR_ENQUEUE is the task executor's enqueue command (infra/executor, on this pod), which `research executor
# enqueue` runs with drive's --graph, --priority and --check (research imports nothing from infra/).
# PYTHONUNBUFFERED: research writes to the loop's log through a pipe, where Python buffers a whole tick's lines until it ends,
# so a running tick showed nothing of where it was.
HELPER='!/data/infra/gh_app_token.sh git'
export PATH=$DIR/ghw:$DIR/bin:/root/.local/bin:$PATH GH_CONFIG_DIR=$DIR/gh-config GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 \
  RESEARCH_LANE=ci RESEARCH_NOTES=/data/steward/research-notes RESEARCH_STORE=/root/drive-store TMPDIR=$DIR/tmp \
  TRAIN_TREES=$FAST/trains \
  RESEARCH_EXECUTOR_ENQUEUE="python3.12 /root/executor/infra/executor.py --db /root/executor/queue.db enqueue"
mkdir -p "$TMPDIR" "$DIR/ghw" "$TRAIN_TREES"
printf '#!/bin/sh\nGH_TOKEN=$(/data/infra/gh_app_token.sh --cached) || exit 1\nexport GH_TOKEN\nexec %s/bin/gh "$@"\n' "$DIR" \
  > "$DIR/ghw/gh.new" && chmod 755 "$DIR/ghw/gh.new" && mv "$DIR/ghw/gh.new" "$DIR/ghw/gh"
# Under the tick lock. `cp -a` keeps each clone's local config (its credential helper and the record merge driver).
fast_clones() {
  mkdir -p "$FAST"
  [ -d "$REPO/.git" ] || cp -a "$DIR/verity" "$REPO" || return 1
  if [ ! -L "$QUEUE" ]; then
    [ -d "$QUEUE/.git" ] && { mv "$QUEUE" "$QUEUE.volume" || return 1; }
    ln -s "$FAST/queue-verity" "$QUEUE" || return 1
  fi
  [ -d "$FAST/queue-verity/.git" ] || [ ! -d "$QUEUE.volume/.git" ] || cp -a "$QUEUE.volume" "$FAST/queue-verity" || return 1
}
# Under the tick lock: a new container's store links its objects/ and attempts/ to their copies on the volume; one already a real
# directory is merged into the volume's copy first (both are write-once, named by content or run id)
store_links() {
  for d in objects attempts; do
    [ -L "$RESEARCH_STORE/$d" ] && continue
    mkdir -p "$RESEARCH_STORE" "$DIR/store/$d" || return 1
    if [ -d "$RESEARCH_STORE/$d" ]; then
      cp -an "$RESEARCH_STORE/$d/." "$DIR/store/$d/" && mv "$RESEARCH_STORE/$d" "$RESEARCH_STORE/$d.old" || return 1
    fi
    ln -s "$DIR/store/$d" "$RESEARCH_STORE/$d" || return 1
    rm -rf "$RESEARCH_STORE/$d.old"
  done
}
# origin's main in $REPO: `merge` puts it and the queue's clone, where `research merge` runs, on branch main, as `research merge`
# refuses any other checkout (the next sync detaches the queue's clone again); the tick and `queue` take a detached one,
# at the commit $DIR/deployed pins (`deploy`) when it names one, else at main's tip, so a landing of drive's own code changes what
# drive runs only once it is deployed.  A pin $REPO can't find fails the tick rather than run another commit.
checkout() {
  fast_clones || return 1
  store_links || return 1
  for c in "$REPO" "$FAST/queue-verity"; do
    [ ! -d "$c/.git" ] || [ "$(git -C "$c" config --get credential.helper)" = "$HELPER" ] \
      || git -C "$c" config credential.helper "$HELPER" || return 1
  done
  git -C "$REPO" fetch -q origin main || return 1
  if [ "$1" = merge ]; then
    git -C "$REPO" checkout -q -f -B main FETCH_HEAD && git -C "$FAST/queue-verity" fetch -q origin main \
      && git -C "$FAST/queue-verity" checkout -q -f -B main FETCH_HEAD
    return
  fi
  pin=FETCH_HEAD
  if [ -e "$DIR/deployed" ]; then
    pin=$(grep -m1 -xE '[0-9a-f]{40}' "$DIR/deployed") \
      || { echo "drive: $DIR/deployed names no commit; \`drive.sh deploy SHA\` or \`drive.sh deploy main\`" >&2; return 1; }
    git -C "$REPO" cat-file -e "$pin^{commit}" 2>/dev/null || git -C "$REPO" fetch -q origin checked || true
  fi
  git -C "$REPO" checkout -q -f --detach "$pin" \
    || { echo "drive: can't check out $pin in $REPO; \`drive.sh deploy main\` runs main's tip" >&2; return 1; }
}
# `drive.sh deploy SHA`: drive runs SHA, a commit `checked` or `main` holds (landed, so reviewed), from the next tick on, whatever
# lands after it, until the next deploy; `drive.sh deploy main` follows main's tip again.
deploy() {
  [ "$1" = main ] && { rm -f "$DIR/deployed"; echo "drive: runs main's tip from the next tick"; return 0; }
  git -C "$REPO" fetch -q origin +refs/heads/main:refs/remotes/origin/main || return 1
  git -C "$REPO" fetch -q origin +refs/heads/checked:refs/remotes/origin/checked 2>/dev/null || true
  sha=$(git -C "$REPO" rev-parse -q --verify "$1^{commit}") \
    || { echo "drive: $1 is no commit $REPO has after fetching main and checked" >&2; return 1; }
  git -C "$REPO" merge-base --is-ancestor "$sha" refs/remotes/origin/main 2>/dev/null \
    || git -C "$REPO" merge-base --is-ancestor "$sha" refs/remotes/origin/checked 2>/dev/null \
    || { echo "drive: $sha is on neither checked nor main: deploy only landed code" >&2; return 1; }
  echo "$sha" > "$DIR/deployed.new" && mv "$DIR/deployed.new" "$DIR/deployed" && echo "drive: runs $sha from the next tick"
}
# Under the tick lock: each train checkout in $TRAIN_TREES unchanged for an hour and no process's cwd, with its log, and the queue
# clone's worktree records of what is gone; and $REPO's packs folded into one once they pass 50: each checkout there fetches the
# blobs it lacks as a pack of its own, and git's auto maintenance never repacks a partial clone's such packs
prune_trees() {
  used=$(for p in /proc/[0-9]*; do readlink "$p/cwd"; done 2>/dev/null | sort -u)
  for t in $(find "$TRAIN_TREES" -mindepth 1 -maxdepth 1 -type d -name 'train-*' -mmin +60); do
    printf '%s\n' "$used" | grep -q "^$t\(/\|$\)" && continue
    [ -n "$(find "$t" -mmin -60 -print -quit 2>/dev/null)" ] && continue
    rm -rf -- "$t" "$t.log"
  done
  git -C "$FAST/queue-verity" worktree prune 2>/dev/null || true
  [ "$(find "$REPO/.git/objects/pack" -name '*.pack' 2>/dev/null | wc -l)" -le 50 ] || git -C "$REPO" repack -a -d -q || true
}
locked() {
  exec 8> "$TICK"; flock 8
  prune_trees
  set -a; . $CONF/r2.env; . $CONF/relay.env; set +a
  [ -n "${RESEARCH_AGENT_URL:-}" ] || RESEARCH_AGENT_URL=$(cat $DIR/agent-url 2>/dev/null); export RESEARCH_AGENT_URL
  GH_TOKEN=$(/data/infra/gh_app_token.sh --cached) || exit 1; export GH_TOKEN
  checkout "$1" || exit 1
  if [ "$1" = merge ]; then cd "$FAST/queue-verity"; else cd "$REPO"; fi || exit 1
  RESEARCH_DRIVE=1 PYTHONPATH=$REPO/tools/research/src exec python3.12 -m research "$@"
}
# `queue baseline` and `queue promote` only fast-forward main, reading the queue's clone and drive's state and writing neither,
# so they run beside a tick: no tick lock, and no checkout, which would change the code a running tick imports from $REPO
unlocked() {
  set -a; . $CONF/r2.env; set +a
  cd "$REPO" && RESEARCH_DRIVE=1 PYTHONPATH=$REPO/tools/research/src exec python3.12 -m research "$@"
}
# Drive's merged trees (the sync's codemod worktrees) go under --preflight-root, on the container disk: under TMPDIR, on the
# network volume, each worktree add and remove takes minutes.
tick() { locked queue drive --config tools/verity/check/drive.toml --preflight-root "$FAST/preflight" $(grep -v '^#' $DIR/args 2>/dev/null) "$@"; }
case ${1:-} in
  start) (exec 8> "$TICK"; flock 8; fast_clones) \
          || { echo "drive: not started: seeding $REPO and the queue's clone from $DIR on the volume failed" >&2; exit 1; }
         (GH_TOKEN=$(/data/infra/gh_app_token.sh) || exit 1; export GH_TOKEN
          git -C "$REPO" fetch -q origin +refs/heads/next:refs/remotes/origin/next \
            && git -C "$REPO" push -q --dry-run origin refs/remotes/origin/next:refs/heads/next) > /dev/null 2>&1 \
          || { echo "drive: not started: gh_app_token.sh failed or a dry-run push of next with its token was refused;" \
                    "drive pushes next and main and edits and closes PRs, so the token needs Contents and Pull requests write on" \
                    "compute-verification/verity" >&2; exit 1; }
         rm -f "$STOP"; echo "drive: started; the loop's next pass (within 30 s) ticks"; exit 0 ;;
  stop) : > "$STOP"; echo "drive: stopped; waiting for a tick in progress to end"; flock "$TICK" true
        echo "drive: no tick running"; exit 0 ;;
  status) [ -f "$STOP" ] && echo "drive: stopped ($STOP)" || echo "drive: started"
          flock -n "$TICK" true && echo "drive: no tick running" || echo "drive: a tick is running"
          echo "drive: runs $(cat "$DIR/deployed" 2>/dev/null || echo "main's tip")"
          tail -n 5 "$LOG" 2>/dev/null; exit 0 ;;
  once) shift; (tick "$@"); exit $? ;;
  queue) case ${2:-} in baseline|promote) (unlocked "$@") ;; *) (locked "$@") ;; esac; exit $? ;;
  merge) (locked "$@"); exit $? ;;
  deploy) deploy "${2:-}"; exit $? ;;
  '') ;;
  *) echo "drive.sh [start|stop|status|once [FLAGS]|queue ARGS|merge ARGS|deploy SHA|main]: with no argument, the loop" >&2; exit 2 ;;
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
