#!/bin/sh
# The pod's lease (research.pods.lease): the dead-man every `research pods create` arms, so a pod is bounded whether or not any VM,
# control pod or guard process is alive.  It ends THIS pod with the pod's own key once the lease has expired and LEASE_GRACE_S has
# passed; nothing else is needed on the pod but a shell and python3 or curl.
#
#   sh lease.sh                    the loop (armed over ssh by `pods create`; also from the start command with --boot-lease)
#   sh lease.sh extend SECONDS BY  push the expiry to at least now + SECONDS (`research pods extend`, `research run --on`); prints it
#   sh lease.sh show               the lease line
#
# The lease file ($LEASE_FILE) is one line, `EPOCH ISO BY`.  The loop re-reads it every tick; a file that is missing or does not
# parse keeps the last expiry read (at boot: $RESEARCH_LEASE_EXPIRES, which `pods create` puts in the pod's environment), so a failed
# read never extends the lease.  At EPOCH the pod guard (pod_guard.sh daemon) ends a pod whose runs all have custody; this loop ends
# it at EPOCH + LEASE_GRACE_S whatever is still unfetched (the grace is the runs' own custody push).
#
# Termination: GraphQL podTerminate with the pod-scoped RUNPOD_API_KEY RunPod puts in the container's first process (measured
# 2026-09-24 through pod_guard.sh: a pod may terminate itself; it may not list or read pods, so `runpodctl get pod` is Unauthorized),
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

lease_read() {
  # the expiry epoch in the lease file, or nothing (missing, unreadable, not a number)
  e=""
  [ -r "$LEASE_FILE" ] && read -r e _rest < "$LEASE_FILE" 2>/dev/null
  case $e in ''|*[!0-9]*) return 1;; esac
  echo "$e"
}

lease_cap() {
  # the latest moment the lease may reach (deadline - margin), or nothing when there is no deadline file
  [ -e "$LEASE_DEADLINE_FILE" ] || return 1
  d=""
  read -r d _rest < "$LEASE_DEADLINE_FILE" 2>/dev/null
  case $d in ''|*[!0-9]*) d=0;; esac
  echo $(( d - LEASE_DEADLINE_MARGIN_S ))
}

lease_env() {
  # $1 from the container's first process, else this shell's environment
  v=$(tr '\0' '\n' < "$LEASE_ENVIRON" 2>/dev/null | sed -n "s/^$1=//p" | head -1)
  if [ -z "$v" ]; then eval "v=\${$1:-}"; fi
  printf '%s' "$v"
}

lease_terminate() {
  me=$1
  if [ -n "${LEASE_TERMINATE_CMD:-}" ]; then $LEASE_TERMINATE_CMD "$me"; return $?; fi
  key=$(lease_env RUNPOD_API_KEY)
  [ -n "$key" ] || { lease_say "no pod-scoped RUNPOD_API_KEY: cannot terminate $me"; return 3; }
  gql=${LEASE_API_GQL:-https://api.runpod.io/graphql}
  rest=${LEASE_API_REST:-https://rest.runpod.io/v1}
  body='{"query": "mutation { podTerminate(input: {podId: \"'"$me"'\"}) }"}'
  if command -v python3 >/dev/null 2>&1; then
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
  elif command -v curl >/dev/null 2>&1; then
    out=$(curl -sS -m 60 -H "Authorization: Bearer $key" -H 'Content-Type: application/json' -d "$body" "$gql" 2>&1)
    case $out in *'"errors"'*|'') echo "GraphQL podTerminate: $out"; rc=1;; *) echo "TERMINATED $me (GraphQL podTerminate, curl)"; rc=0;; esac
  else
    rc=1
  fi
  [ "$rc" -eq 0 ] && return 0
  if command -v runpodctl >/dev/null 2>&1; then
    RUNPOD_API_KEY=$key runpodctl remove pod "$me" && { echo "TERMINATED $me (runpodctl)"; return 0; }
  fi
  return 1
}

lease_extend() {
  # lease_extend SECONDS BY: the expiry becomes max(its value, now + SECONDS); prints the lease line
  secs=${1:?seconds}; by=${2:-hand}
  case $secs in ''|*[!0-9]*) echo "lease: extend wants whole seconds, got '$secs'" >&2; return 2;; esac
  want=$(( $(lease_now) + secs ))
  cur=$(lease_read) || cur=${RESEARCH_LEASE_EXPIRES:-0}
  case $cur in ''|*[!0-9]*) cur=0;; esac
  if cap=$(lease_cap); then
    if [ "$cur" -gt "$cap" ]; then cur=0; want=$cap; fi
    if [ "$want" -ge "$cap" ]; then want=$cap; by="$by(clamped-to-deadline)"; fi
  fi
  if [ "$want" -gt "$cur" ]; then
    mkdir -p "$(dirname "$LEASE_FILE")" || return 1
    printf '%s %s %s\n' "$want" "$(lease_iso "$want")" "$by" > "$LEASE_FILE.tmp.$$" && mv -f "$LEASE_FILE.tmp.$$" "$LEASE_FILE" || return 1
    lease_say "extended to $(lease_iso "$want") by $by"
  fi
  cat "$LEASE_FILE" 2>/dev/null || printf '%s %s %s\n' "$cur" "$(lease_iso "$cur")" "(environment)"
}

lease_claim() {
  # one loop per pod: a live loop named in the pid file keeps it
  old=$(cat "$LEASE_PID" 2>/dev/null)
  if [ -n "$old" ] && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null; then
    case $(tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null) in *lease.sh*) return 1;; esac
  fi
  echo "$$" > "$LEASE_PID" 2>/dev/null
  return 0
}

lease_loop() {
  mkdir -p "$LEASE_DIR" 2>/dev/null
  if [ "${LEASE_ONCE:-0}" != 1 ] && ! lease_claim; then
    lease_say "another lease loop ($(cat "$LEASE_PID" 2>/dev/null)) holds this pod; exiting"
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
  while :; do
    if e=$(lease_read); then exp=$e; fi
    now=$(lease_now)
    due=0
    if [ -n "$exp" ] && [ "$now" -ge $((exp + LEASE_GRACE_S)) ]; then due=1; fi
    if cap=$(lease_cap) && [ "$now" -ge "$cap" ]; then due=1; exp=$cap; fi
    if [ "$due" = 1 ]; then
      if [ -z "$me" ]; then
        lease_say "lease expired $(lease_iso "$exp") but no RUNPOD_POD_ID: cannot terminate"
        rc=2
      else
        lease_say "lease expired $(lease_iso "$exp") + ${LEASE_GRACE_S}s grace: terminating $me"
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
  extend) shift; lease_extend "$@" ;;
  show) cat "$LEASE_FILE" 2>/dev/null || { echo "no lease file at $LEASE_FILE" >&2; exit 1; } ;;
  *) echo "usage: lease.sh [loop | extend SECONDS BY | show]" >&2; exit 2 ;;
esac
