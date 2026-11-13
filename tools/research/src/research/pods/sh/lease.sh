#!/bin/sh
# The pod's lease (research.pods.lease): the dead-man every `research pods create` arms, so a pod is bounded whether or not any VM,
# control pod or guard process is alive.  It ends THIS pod with the pod's own key once the lease has expired and LEASE_GRACE_S has
# passed; nothing else is needed on the pod but a shell and python3 or curl.
#
#   sh lease.sh                    the loop (the start command with --boot-lease, a machine's own service)
#   sh lease.sh start              the loop, detached, unless the one the pid file names runs; prints its pid (`pods create`, over ssh)
#   sh lease.sh pid                the pid of the running loop the pid file names; exit 1 when it names none
#   sh lease.sh extend SECONDS BY  push the expiry to at least now + SECONDS (`research pods extend`, `research run --on`); prints it
#   sh lease.sh show               the lease line
#
# One loop per pod.  The pid file ($LEASE_PID) is one line, `PID BIRTH`: the loop's pid and when that process was forked
# (lease_birth), which together name one process for its whole life.  While the process it names runs, `start` starts no other
# and a second loop exits; a file that names no running process (none, a zombie, a later process given the pid, a pid with no
# BIRTH) names no loop.  The command line is not read: it is empty while a process is inside execve, which a loop started through
# setsid and nohup is three times.
#
# The lease file ($LEASE_FILE) is one line, `EPOCH ISO BY`.  The loop re-reads it every tick; a file that is missing or does not
# parse keeps the last expiry read (at boot: $RESEARCH_LEASE_EXPIRES, which `pods create` puts in the pod's environment), so a failed
# read never extends the lease.  At EPOCH the pod guard (pod_guard.sh daemon) ends a pod whose runs all have custody; this loop ends
# it at EPOCH + LEASE_GRACE_S whatever is still unfetched (the grace is the runs' own custody push).
#
# A read error is neither a missing file nor an expiry.  A lease or deadline file that is there and cannot be read (lease_first's
# exit 2) is logged every tick, and the loop keeps what it last read of that file: the expiry, the deadline.  With nothing read to
# keep (no expiry since the loop started, a deadline it never read) the loop ends the machine once that has lasted LEASE_GRACE_S, so
# no fault carries a machine past its lease for ever.  `extend` has nothing it last read: it says the error on stderr and takes a
# lease it cannot read as missing (it writes a new one) and a deadline it cannot read as a deadline of 0 (it extends nothing).
# What the loop decides on (the two files, the pod's id and key, the tool that terminates) opens no other path: no redirect to
# the null device, which a machine can lose.
#
# Termination: GraphQL podTerminate with the pod-scoped RUNPOD_API_KEY RunPod puts in the container's first process (as measured
# through pod_guard.sh: a pod may terminate itself; it may not list or read pods, so `runpodctl get pod` is Unauthorized),
# then REST DELETE, then `runpodctl remove pod`.  The account key is never on a pod.
# Test seams: LEASE_NOW (epoch), LEASE_ONCE=1 (one tick, exit 0 terminated / 1 not due / 2 failed), LEASE_TERMINATE_CMD (run with the
# pod id as $1 instead of the API), LEASE_API_GQL / LEASE_API_REST, LEASE_ENVIRON (default /proc/1/environ).
# A machine that is not a RunPod pod (an ssh machine: pods/nebius/) names itself with LEASE_MACHINE_ID and ends itself through
# LEASE_TERMINATE_CMD, which is then the production path, not a seam.
#
# Hard deadline: when $LEASE_DEADLINE_FILE exists (root's, `EPOCH ISO BY`: the end of the spend authorization), nothing carries the
# machine past it.  `extend` clamps the expiry to EPOCH - LEASE_DEADLINE_MARGIN_S, and the loop ends the machine from that moment
# whatever the lease file says (no grace), so it has stopped by EPOCH.  A deadline file that does not parse is a deadline of 0.

LEASE_DIR=${LEASE_DIR:-/root/.research}
LEASE_DEADLINE_FILE=${LEASE_DEADLINE_FILE:-/etc/research/deadline}
LEASE_DEADLINE_MARGIN_S=${LEASE_DEADLINE_MARGIN_S:-300}
LEASE_FILE=${LEASE_FILE:-$LEASE_DIR/lease}
LEASE_LOG=${LEASE_LOG:-$LEASE_DIR/lease.log}
LEASE_PID=${LEASE_PID:-$LEASE_DIR/lease.pid}
LEASE_GRACE_S=${LEASE_GRACE_S:-900}
LEASE_TICK_S=${LEASE_TICK_S:-60}
LEASE_ENVIRON=${LEASE_ENVIRON:-/proc/1/environ}

lease_now() { if [ -n "${LEASE_NOW:-}" ]; then echo "$LEASE_NOW"; else date +%s; fi; }
lease_iso() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "@$1"; }
lease_say() { echo "$(lease_iso "$(lease_now)") lease pid $$: $*" >> "$LEASE_LOG" 2>/dev/null || echo "lease: $*" >&2; }

lease_first() {
  # lease_first FILE: the first field of FILE's first line (nothing for an empty file).  Exit 1 when there is no such file; exit 2
  # when the file is there and cannot be read, with the reader's own message in place of the field.  It opens FILE and nothing else.
  [ -e "$1" ] || return 1
  line=$(head -n 1 "$1" 2>&1) || { printf '%s\n' "$line"; return 2; }
  set -f
  # shellcheck disable=SC2086
  set -- $line
  set +f
  printf '%s\n' "${1:-}"
}

lease_read() {
  # the expiry epoch in the lease file.  Exit 1: no lease file, or one that names no epoch; exit 2: a read error (lease_first's)
  e=$(lease_first "$LEASE_FILE") || { st=$?; printf '%s\n' "$e"; return "$st"; }
  case $e in ''|*[!0-9]*) return 1;; esac
  echo "$e"
}

lease_cap() {
  # the latest moment the lease may reach (deadline - margin).  Exit 1: no deadline file; exit 2: a read error (lease_first's)
  d=$(lease_first "$LEASE_DEADLINE_FILE") || { st=$?; printf '%s\n' "$d"; return "$st"; }
  case $d in ''|*[!0-9]*) d=0;; esac
  echo $(( d - LEASE_DEADLINE_MARGIN_S ))
}

lease_env() {
  # $1 from the container's first process, else this shell's environment
  v=""
  [ -r "$LEASE_ENVIRON" ] && v=$(tr '\0' '\n' < "$LEASE_ENVIRON" | sed -n "s/^$1=//p" | head -1)
  if [ -z "$v" ]; then eval "v=\${$1:-}"; fi
  printf '%s' "$v"
}

lease_has() { [ -n "$(command -v "$1")" ]; }

lease_terminate() {
  me=$1
  if [ -n "${LEASE_TERMINATE_CMD:-}" ]; then $LEASE_TERMINATE_CMD "$me"; return $?; fi
  key=$(lease_env RUNPOD_API_KEY)
  [ -n "$key" ] || { lease_say "no pod-scoped RUNPOD_API_KEY: cannot terminate $me"; return 3; }
  gql=${LEASE_API_GQL:-https://api.runpod.io/graphql}
  rest=${LEASE_API_REST:-https://rest.runpod.io/v1}
  body='{"query": "mutation { podTerminate(input: {podId: \"'"$me"'\"}) }"}'
  if lease_has python3; then
    LEASE_KEY="$key" LEASE_BODY="$body" python3 - research-pod-lease "$gql" "$rest" "$me" <<'PY'
import os, sys, urllib.error, urllib.request
gql, rest, me = sys.argv[2:5]
key = os.environ.pop("LEASE_KEY")
hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "research-pod-lease"}
try:
    with urllib.request.urlopen(urllib.request.Request(gql, data=os.environ["LEASE_BODY"].encode(), headers=hdr, method="POST"), timeout=60) as r:
        body = r.read()[:300].decode(errors="replace")
    if '"errors"' not in body:
        print(f"TERMINATED {me} (GraphQL podTerminate)"); sys.exit(0)
    print(f"GraphQL podTerminate: {body[:200]}")
except Exception as e:  # noqa: BLE001
    print(f"GraphQL podTerminate {type(e).__name__}: {e}")
try:
    with urllib.request.urlopen(urllib.request.Request(f"{rest}/pods/{me}", headers=hdr, method="DELETE"), timeout=60) as r:
        print(f"TERMINATED {me} (REST DELETE, HTTP {r.status})"); sys.exit(0)
except Exception as e:  # noqa: BLE001
    print(f"REST DELETE {type(e).__name__}: {e}")
sys.exit(1)
PY
    rc=$?
  elif lease_has curl; then
    out=$(curl -sS -m 60 -H "Authorization: Bearer $key" -H 'Content-Type: application/json' -d "$body" "$gql" 2>&1)
    case $out in *'"errors"'*|'') echo "GraphQL podTerminate: $out"; rc=1;; *) echo "TERMINATED $me (GraphQL podTerminate, curl)"; rc=0;; esac
  else
    rc=1
  fi
  [ "$rc" -eq 0 ] && return 0
  if lease_has runpodctl; then
    RUNPOD_API_KEY=$key runpodctl remove pod "$me" && { echo "TERMINATED $me (runpodctl)"; return 0; }
  fi
  return 1
}

lease_extend() {
  # lease_extend SECONDS BY: the expiry becomes max(its value, now + SECONDS); prints the lease line
  secs=${1:?seconds}; by=${2:-hand}
  case $secs in ''|*[!0-9]*) echo "lease: extend wants whole seconds, got '$secs'" >&2; return 2;; esac
  want=$(( $(lease_now) + secs ))
  cur=$(lease_read); st=$?
  if [ "$st" = 2 ]; then echo "lease: cannot read $LEASE_FILE ($cur): taken as missing" >&2; fi
  [ "$st" = 0 ] || cur=${RESEARCH_LEASE_EXPIRES:-0}
  case $cur in ''|*[!0-9]*) cur=0;; esac
  latest=$(lease_cap); st=$?
  if [ "$st" = 2 ]; then
    echo "lease: cannot read $LEASE_DEADLINE_FILE ($latest): taken as a deadline of 0, so nothing is extended" >&2
    latest=$(( 0 - LEASE_DEADLINE_MARGIN_S ))
  fi
  if [ "$st" != 1 ]; then
    if [ "$cur" -gt "$latest" ]; then cur=0; want=$latest; fi
    if [ "$want" -ge "$latest" ]; then want=$latest; by="$by(clamped-to-deadline)"; fi
  fi
  if [ "$want" -gt "$cur" ]; then
    mkdir -p "$(dirname "$LEASE_FILE")" || return 1
    printf '%s %s %s\n' "$want" "$(lease_iso "$want")" "$by" > "$LEASE_FILE.tmp.$$" && mv -f "$LEASE_FILE.tmp.$$" "$LEASE_FILE" || return 1
    lease_say "extended to $(lease_iso "$want") by $by"
  fi
  cat "$LEASE_FILE" 2>/dev/null || printf '%s %s %s\n' "$cur" "$(lease_iso "$cur")" "(environment)"
}

lease_birth() {
  # lease_birth PID: when the running process PID was forked, `TICKS@BOOT`, or nothing (no such process, or a zombie).  TICKS is
  # field 22 of /proc/PID/stat (clock ticks since boot, set at fork and unchanged by execve), counted after the name's closing `)`
  # since a name may hold spaces; BOOT is the boot's id, since ticks start over at each boot and a pid file can outlive one.
  stat=$(sed -n '$s/.*) //p' "/proc/$1/stat" 2>/dev/null)
  case $stat in ''|[ZXx]' '*) return 1;; esac
  echo "$(echo "$stat" | cut -d' ' -f20)@$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
}

lease_holder() {
  # the pid of the running loop the pid file names, or nothing
  pid=""; born=""
  [ -r "$LEASE_PID" ] && read -r pid born _rest < "$LEASE_PID" 2>/dev/null
  [ -n "$born" ] && [ "$(lease_birth "$pid")" = "$born" ] && echo "$pid"
}

lease_hold() {
  # lease_hold PID: name PID in the pid file, written beside it and moved into place so that no reader finds it empty
  { printf '%s %s\n' "$1" "$(lease_birth "$1")" > "$LEASE_PID.tmp.$$" && mv -f "$LEASE_PID.tmp.$$" "$LEASE_PID"; } 2>/dev/null
}

lease_claim() {
  # one loop per pod: this one runs unless the pid file names another that does
  if old=$(lease_holder); then [ "$old" = "$$" ]; else lease_hold "$$"; return 0; fi
}

lease_start() {
  # the loop, detached and named in the pid file before this returns, unless the one the file names runs; prints its pid
  if ! pid=$(lease_holder); then
    setsid nohup sh "$0" loop >> "$LEASE_LOG" 2>&1 < /dev/null &
    pid=$!
    lease_hold "$pid"
  fi
  echo "$pid"
}

lease_loop() {
  mkdir -p "$LEASE_DIR"
  if [ "${LEASE_ONCE:-0}" != 1 ] && ! lease_claim; then
    lease_say "another lease loop ($old) holds this pod; exiting"
    return 0
  fi
  exp=${RESEARCH_LEASE_EXPIRES:-}
  case $exp in *[!0-9]*) exp="";; esac
  me=$(lease_env RUNPOD_POD_ID)
  [ -n "$me" ] || me=${LEASE_MACHINE_ID:-}
  if e=$(lease_read); then exp=$e; fi
  if [ "${LEASE_ONCE:-0}" != 1 ]; then
    if [ -n "$exp" ]; then shown=$(lease_iso "$exp"); else shown="unknown (no lease file, no RESEARCH_LEASE_EXPIRES)"; fi
    lease_say "armed: pod ${me:-?}, expiry $shown, grace ${LEASE_GRACE_S}s"
  fi
  latest=""; since=""
  while :; do
    now=$(lease_now)
    bad=""; blind=0
    e=$(lease_read); st=$?
    if [ "$st" = 0 ]; then exp=$e
    elif [ "$st" = 2 ]; then bad="$LEASE_FILE ($e)"; [ -n "$exp" ] || blind=1
    fi
    c=$(lease_cap); st=$?
    if [ "$st" = 0 ]; then latest=$c
    elif [ "$st" = 1 ]; then latest=""
    else bad="${bad:+$bad and }$LEASE_DEADLINE_FILE ($c)"; [ -n "$latest" ] || blind=1
    fi
    if [ "$blind" = 1 ]; then [ -n "$since" ] || since=$now; else since=""; fi
    if [ -n "$since" ]; then
      ends=$(lease_iso $((since + LEASE_GRACE_S)))
      lease_say "cannot read $bad: not an expiry, but nothing was read to keep; the machine ends at $ends unless a read succeeds"
    elif [ -n "$bad" ]; then
      lease_say "cannot read $bad: not an expiry; keeping what was last read"
    fi
    due=""
    if [ -n "$exp" ] && [ "$now" -ge $((exp + LEASE_GRACE_S)) ]; then due=lease; fi
    if [ -n "$latest" ] && [ "$now" -ge "$latest" ]; then due=lease; exp=$latest; fi
    if [ -z "$due" ] && [ -n "$since" ] && [ "$now" -ge $((since + LEASE_GRACE_S)) ]; then due=unread; fi
    if [ -n "$due" ]; then
      if [ "$due" = lease ]; then why="lease expired $(lease_iso "$exp")"; else why="read errors with nothing to keep since $(lease_iso "$since")"; fi
      if [ -z "$me" ]; then
        lease_say "$why but no RUNPOD_POD_ID: cannot terminate"
        rc=2
      else
        lease_say "$why + ${LEASE_GRACE_S}s grace: terminating $me"
        out=$(lease_terminate "$me" 2>&1); rc=$?
        lease_say "$out"
        [ "$rc" -eq 0 ] && return 0
        rc=2
      fi
    else
      rc=1
    fi
    [ "${LEASE_ONCE:-0}" = 1 ] && return "$rc"
    sleep "$LEASE_TICK_S"
  done
}

case ${1:-loop} in
  loop) lease_loop ;;
  start) lease_start ;;
  pid) lease_holder ;;
  extend) shift; lease_extend "$@" ;;
  show) cat "$LEASE_FILE" 2>/dev/null || { echo "no lease file at $LEASE_FILE" >&2; exit 1; } ;;
  *) echo "usage: lease.sh [loop | start | pid | extend SECONDS BY | show]" >&2; exit 2 ;;
esac
