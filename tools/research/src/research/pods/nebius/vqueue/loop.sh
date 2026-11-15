#!/bin/bash
# The verified queue's loop on a node, as vy-vqueue under vy-vqueue.service: one tick of `research queue vqueue --once` a pass,
# run from ci's pinned tree in the loop's own clone.
#
#   loop.sh run             the loop: a tick, the sweep, VQ_INTERVAL seconds, again; SIGTERM ends it after the tick in progress
#   loop.sh once [FLAG ..]  one tick now, in the mode below, with FLAGs last
#   loop.sh git get         git's credential helper for the clone: the App's token of the moment
#
# Everything it keeps is under its home (VQ_HOME, the unit's StateDirectory), on the node's own disk:
#   mode      dry-run (also when the file is absent), shadow or live
#   live/     the loop's state: decisions.jsonl (the log every tick derives its candidates from), state.json, vqueue.lock, and
#             ci's files: args (the loop's flags, one a line, `#` comments), priority, agent-url, frozen, stopped, and the lists
#             every mode reads: deselect, keep-runs, wide, heavy and pins (LISTS)
#   shadow/   a shadow's own log, with its own args and stopped
#   verity/   the loop's clone;  src/  ci's pinned tree, whose research package and train.sh run, behind origin/main or
#             not (RESEARCH_AT_CHECKOUT=1: research re-executes from origin/main's code otherwise);  tmp/, trains/  scratch
# and what research keeps under a home: .research/ (the store's cache, the records of the runs it launched, the notes clone with
# machines.d) and .runpod/ssh/ (its key to the machines its checks run on).
#
# Modes. dry-run adds --dry-run and reads live/: the tick prints its snapshot and what it would do and writes nothing (no ref,
# log line, label, message or run), so it runs beside another host's loop on a copy of that loop's log. shadow adds --shadow and
# keeps its log in shadow/: it builds candidates (pushing queue/* to origin) and launches their checks on its args' --on
# machines, and only logs each advance, eject and message, so `main` never moves; it costs those machines. live is the loop.
# It is refused while live/decisions.jsonl is missing or empty: a loop with no log rebuilds every candidate in flight. While
# <the mode's directory>/stopped exists no tick starts. mode() alone decides the mode, and with it RESEARCH_DRIVE=1, without
# which the loop never moves `main` (merge.on_drive): live's tick gets it, and dry-run's and shadow's never, whatever the unit's
# environment or a credential file says.
#
# The node's paths follow ci's args on the command line, so a flag of args naming another host's file never wins: the log, the
# lock, the five lists (each read as empty while it is missing, as on any host; a tick names the missing ones in one line) and
# --graph2, graph2.sh beside this script. The steps after a landing publish the Lean cache with lean_publish.sh beside it
# (RESEARCH_LEAN_PUBLISH; a credential file may set it empty to turn the publish off).
#
# Credentials, none in this file or its unit: VQ_CRED/*.env (root's, readable by the group: the store's key, the Slack token)
# are read for each tick as data, never run as shell (creds), and GH_TOKEN is the token vy-vqueue-token.timer keeps in VQ_TOKEN, which root mints from the App's
# key: this user can't read the key. gh on PATH is bin/gh, which reads that file for each call, and the clone's credential
# helper is this script. A token file older than 50 min means the timer failed: the tick is skipped rather than run on a token
# about to expire.
#
# A launch outlives its tick (train.sh, then check.py attached to its run), and the loop judges it by its pid on this host, so
# the unit restarts with KillMode=process and scratch stays under the home, never a private /tmp.
set -u
H=${VQ_HOME:-/var/lib/vy-vqueue} CRED=${VQ_CRED:-/etc/vy/vqueue} TOKEN=${VQ_TOKEN:-/run/vy-vqueue/token}
PY=${VQ_PYTHON:-python3.12} INTERVAL=${VQ_INTERVAL:-60}
SELF=$(readlink -f "$0")
LIB=$(dirname "$SELF")
CLONE=$H/verity CODE=$H/src
CONFIG=78     # what the node holds can't run a tick: the unit stays failed until it is put right (RestartPreventExitStatus)
SKIP=75       # this tick can't run, the next may: the loop goes on
# each list's flag and its file under live/
LISTS="--deselect=deselect --keep-runs=keep-runs --wide-file=wide --heavy-file=heavy --pins-file=pins"
# what a credential file may set: the store's remote, the Slack token, and the names graph2.sh and lean_publish.sh read
KEYS="AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY R2_ACCOUNT_ID R2_BUCKET R2_ENDPOINT SLACK_BOT_TOKEN VQ_GRAPH2_NODE VQ_LEAN_NODE \
RESEARCH_LEAN_PUBLISH"

if [ "${1:-}" = git ]; then
  cat > /dev/null
  [ "${2:-}" = get ] || exit 0
  t=$(cat "$TOKEN") || exit 1
  printf 'username=x-access-token\npassword=%s\n' "$t"
  exit 0
fi

export HOME=$H PATH="$LIB/bin:/usr/local/bin:/usr/bin:/bin" GIT_TERMINAL_PROMPT=0 LC_ALL=C.UTF-8 \
  PYTHONUNBUFFERED=1 RESEARCH_LANE=ci TMPDIR=$H/tmp TRAIN_TREES=$H/trains RESEARCH_LEAN_PUBLISH=$LIB/lean_publish.sh \
  RESEARCH_AT_CHECKOUT=1

say() { echo "vy-vqueue: $*" >&2; }

# Sets mode, dir (the mode's directory), flag and drive (1: the tick may move `main`).
mode() {
  mode=dry-run drive=
  [ ! -s "$H/mode" ] || read -r mode _ < "$H/mode"
  case $mode in
    dry-run) dir=$H/live flag=--dry-run ;;
    shadow)  dir=$H/shadow flag=--shadow ;;
    live)    dir=$H/live flag= drive=1 ;;
    *) say "$H/mode says '$mode': write dry-run, shadow or live"; return $CONFIG ;;
  esac
}

# Sets listed, the lists' flags with the node's files, and says the missing ones in one line.
lists() {
  local l missing=
  listed=()
  for l in $LISTS; do
    listed+=("${l%%=*}" "$H/live/${l#*=}")
    [ -e "$H/live/${l#*=}" ] || missing="$missing ${l#*=}"
  done
  [ -z "$missing" ] || say "no$missing in $H/live: the loop reads each as empty; copy ci's from the host the loop ran on"
}

# A value sh reads as its own text: in single quotes, in double quotes with nothing sh expands, or bare with no character
# sh treats apart. Any other is refused rather than read one way here and another by a shell sourcing the same file.
SQ="^'([^']*)'\$" DQ='^"([^"$`\\]*)"$' PLAIN='^([A-Za-z0-9_./:@%+=,-]*)$'

# Exports what VQ_CRED/*.env set, one KEY=VALUE a line (blank and `#` lines and `export ` before a key allowed), as data: no
# line runs, and none can set what this script decides, since a key not in KEYS refuses the tick, naming the file, the line
# and the key, never a value.
creds() {
  local f n line k v
  for f in "$CRED"/*.env; do
    [ -f "$f" ] || continue
    n=0
    while IFS= read -r line || [ -n "$line" ]; do
      n=$((n + 1))
      line=${line#"${line%%[![:space:]]*}"}
      case $line in ''|'#'*) continue ;; esac
      line=${line#export }
      k=${line%%=*} v=${line#*=}
      if [ "$k" = "$line" ] || ! [[ $k =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        say "$f line $n is not KEY=VALUE: write one key the loop reads a line ($KEYS)"
        return $CONFIG
      fi
      case " $KEYS " in
        *" $k "*) ;;
        *) say "$f line $n sets $k, which the loop doesn't read: remove it (the keys it reads: $KEYS)"; return $CONFIG ;;
      esac
      if [[ $v =~ $SQ ]] || [[ $v =~ $DQ ]] || [[ $v =~ $PLAIN ]]; then
        export "$k=${BASH_REMATCH[1]}"
      else
        say "$f line $n: sh would read $k's value as other than its text: put the value in single quotes"
        return $CONFIG
      fi
    done < "$f"
  done
}

now() { date +%s; }

# The App's token, written within the last 50 min.
fresh() {
  [ -s "$TOKEN" ] && [ $(( $(now) - $(stat -c %Y "$TOKEN") )) -lt 3000 ]
}

# One tick in the mode's directory; FLAGs follow.
tick() {
  mode || return
  if [ "$mode" = live ] && [ ! -s "$dir/decisions.jsonl" ]; then
    say "live needs the loop's log at $dir/decisions.jsonl: copy it from the host the loop ran on, or write dry-run to $H/mode"
    return $CONFIG
  fi
  [ -d "$CODE/tools/research/src/research" ] || { say "no pinned tree at $CODE: check ci's pin out there"; return $CONFIG; }
  [ -d "$CLONE/.git" ] || { say "no clone at $CLONE: copy the loop's clone there (its config holds the merge drivers)"; return $CONFIG; }
  [ -f "$dir/args" ] || { say "no $dir/args: the loop's flags, one a line (--gate, --on, --batch, --depth, --lean)"; return $CONFIG; }
  fresh || { say "the App's token at $TOKEN is missing or over 50 min old: systemctl status vy-vqueue-token.service"; return $SKIP; }
  mkdir -p "$TMPDIR" "$TRAIN_TREES" || return $CONFIG
  [ "$(git -C "$CLONE" config --local --get credential.helper)" = "!$SELF git" ] \
    || git -C "$CLONE" config --local credential.helper "!$SELF git" || return $CONFIG
  (
    creds || exit $CONFIG
    if [ "$drive" = 1 ]; then export RESEARCH_DRIVE=1; else unset RESEARCH_DRIVE; fi
    [ ! -s "$H/live/agent-url" ] || { read -r RESEARCH_AGENT_URL _ < "$H/live/agent-url"; export RESEARCH_AGENT_URL; }
    GH_TOKEN=$(cat "$TOKEN") || exit $SKIP
    export GH_TOKEN
    lists
    cd "$CLONE" || exit $CONFIG
    set -f      # args is words, a flag or a value each, never a pattern
    # shellcheck disable=SC2046
    PYTHONPATH=$CODE/tools/research/src exec "$PY" -m research queue vqueue $(grep -v '^[[:space:]]*#' "$dir/args") \
      --repo "$CLONE" --log "$dir/decisions.jsonl" --state "$dir/state.json" --lock "$dir/vqueue.lock" \
      --priority-file "$H/live/priority" --frozen "$dir/frozen" "${listed[@]}" --graph2 "$LIB/graph2.sh" --once $flag "$@"
  )
}

# What a launch leaves behind (train.sh's tree and its log, check.py's Lean deps, the tree of a job of the steps after a landing
# that was killed: `research-after-land-*`), 40 min after its last change, and never
# while a process works in it or under it (a log: in its tree). train.sh starts check.py in the tree, on the venv under it,
# where it can stay for as long as its run, and the tree's own time stays at the launch. The processes seen are this user's,
# which every launch of the loop's is.
sweep() {
  local before busy p changed at
  before=$(( $(now) - 2400 ))
  busy=$'\n'$(readlink /proc/[0-9]*/cwd 2>/dev/null)$'\n'
  for p in "$TRAIN_TREES"/train-* "$TRAIN_TREES"/research-after-land-* "$TMPDIR"/train-* "$TMPDIR"/check-lean-deps-*; do
    changed=$(stat -c %Y "$p") || continue
    [ "$changed" -lt "$before" ] || continue
    at=$(readlink -f "${p%.log}") || continue
    case $busy in *$'\n'"$at"$'\n'*|*$'\n'"$at"/*) continue ;; esac
    rm -rf "$p"
  done 2>/dev/null
  git -C "$CLONE" worktree prune 2>/dev/null
}

run() {
  stop=
  trap 'stop=1' TERM INT
  while [ -z "$stop" ]; do
    mode || exit
    if [ ! -e "$dir/stopped" ]; then
      tick
      rc=$?
      echo "vy-vqueue: tick rc=$rc mode=$mode pin=$(git -C "$CODE" rev-parse --short=12 HEAD 2>/dev/null)"
      [ "$rc" != "$CONFIG" ] || exit $CONFIG
      sweep
    fi
    [ -z "$stop" ] || break
    sleep "$INTERVAL" &
    wait $!
    kill $! 2>/dev/null
  done
  say "stopped after its tick"
}

case ${1:-} in
  run) run ;;
  once) shift; tick "$@" ;;
  *) say "usage: loop.sh run | once [FLAG ..] | git get"; exit 2 ;;
esac
