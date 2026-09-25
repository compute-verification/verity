#!/usr/bin/env bash
# Pod-side guard (integrator, 2026-09-13, D55) — sourced ON THE POD by every job launcher.  Nothing here depends on the laptop or on
# the agent that launched the job being alive.  Needs the RUNPOD_POD_ID and RUNPOD_API_KEY RunPod injects into every pod's first process;
# only pods an old launcher prepared still carry /root/.runpod/config.toml + /root/dm/runpod.py (the account key), used as a fallback.
#
#   source /root/dm/pod_guard.sh
#   guard_start 'e2e.interior.producer|run_tiny1_pod|reverify' 20     # terminate THIS pod after 20 min with no process matching the pattern
#   ... the job chain (set -e) ...
#   guard_finish "reason"                                                 # sync outputs are already on /workspace (the volume); terminate THIS pod
#
# Rules it implements:
#   * idle shutdown  — guard_start's loop: when no process matches JOB_PATTERN for IDLE_MIN consecutive minutes, the pod terminates itself
#                      (a pod is never held idle waiting for a laptop);
#   * completion     — guard_finish terminates the pod at the end of the chain, success or failure (the chain's `trap` calls it);
#   * heartbeat      — /workspace/<who>/guard/<pod>.hb is touched every minute so the state can be read from the volume by anyone;
#   * durability     — (F7) termination is refused while any attempt registered with `guard_require_preserved` is not PRESERVED on the
#                      remote (`research data preserved`); one `research data push` repair is tried first; GUARD_FORCE_TERMINATE=1 overrides.
#   * credential     — (2026-09-24) the pod terminates itself with its OWN key: RUNPOD_API_KEY, which RunPod injects into the container's
#                      first process (/proc/1/environ; not the account key).  Measured on a cpu3c pod: it can read nothing (REST GET
#                      /pods and /pods/{self} 403, GraphQL myself Unauthorized) and REST DELETE /pods/{self} is 403, but GraphQL
#                      podTerminate of its own pod succeeds (guard_terminate_pod tries REST, then GraphQL, then runpodctl).  The account
#                      key's /root/dm/runpod.py path is only a fallback for pods an old launcher copied it to; `research run --on` never does.
# The daemon-side per-pod age ceiling (budget_cap.py --pod-max-hours) and the cap/rate/deadline on the controller pod remain the outer layers.
#
# `research run --on` (the machine side, research.remote.ensure_guard) runs this file as ONE daemon per pod: `guard_daemon`, at the end.

GUARD_ENVIRON=${GUARD_ENVIRON:-/proc/1/environ}      # RunPod's injected environment (ssh sessions do not inherit it)
GUARD_LOGF=${GUARD_LOGF:-/root/dm/guard.log}

guard_self_id() {
  # RunPod sets RUNPOD_POD_ID for the container's main process; ssh sessions do not inherit it -> read /proc/1/environ, then /root/dm/POD_ID.
  if [ -n "${RUNPOD_POD_ID:-}" ]; then echo "$RUNPOD_POD_ID"; return; fi
  tr '\0' '\n' < "$GUARD_ENVIRON" 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' | head -1 | grep . && return
  cat /root/dm/POD_ID 2>/dev/null
}

guard_pod_key() {
  # the pod-scoped key from RunPod's injected environment only (never a shell's RUNPOD_API_KEY, never a file); printed into a pipe only
  tr '\0' '\n' < "$GUARD_ENVIRON" 2>/dev/null | sed -n 's/^RUNPOD_API_KEY=//p' | head -1 | tr -d '\n'
}

guard_terminate_pod() {
  # terminate pod $1 with the pod-scoped key: REST DELETE /pods/{id}, else GraphQL podTerminate, else runpodctl.  0 = accepted.
  # The key is read from $GUARD_ENVIRON by the python below itself (python's stdin is its script; the key is in no argv or env).
  # GUARD_TERMINATE_CMD replaces all of it: it is run with the pod id as $1.  GUARD_API_REST / GUARD_API_GQL point at a fake API (tests).
  local me=${1:?pod id} rc
  if [ -n "${GUARD_TERMINATE_CMD:-}" ]; then $GUARD_TERMINATE_CMD "$me"; return $?; fi
  GUARD_POD="$me" GUARD_ENVIRON="$GUARD_ENVIRON" "${GUARD_PY:-python3}" - research-pod-guard <<'PY'
import json, os, sys, urllib.error, urllib.request
me = os.environ["GUARD_POD"]
try:
    env = open(os.environ["GUARD_ENVIRON"], "rb").read().split(b"\0")
except OSError:
    env = []
key = next((kv.split(b"=", 1)[1].decode() for kv in env if kv.startswith(b"RUNPOD_API_KEY=")), "").strip()
if not key:
    print("no pod-scoped RUNPOD_API_KEY in the pod's environment"); sys.exit(3)
rest = os.environ.get("GUARD_API_REST", "https://rest.runpod.io/v1")
gql = os.environ.get("GUARD_API_GQL", "https://api.runpod.io/graphql")
def call(method, url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "research-pod-guard"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()[:300].decode(errors="replace")
try:
    st, _ = call("DELETE", f"{rest}/pods/{me}")
    print(f"TERMINATED {me} (REST DELETE, HTTP {st})"); sys.exit(0)
except urllib.error.HTTPError as e:
    print(f"REST DELETE HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")
except Exception as e:  # noqa: BLE001
    print(f"REST DELETE {type(e).__name__}: {e}")
try:
    st, body = call("POST", gql, {"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % me})
    if '"errors"' not in body:
        print(f"TERMINATED {me} (GraphQL podTerminate, HTTP {st})"); sys.exit(0)
    print(f"GraphQL podTerminate HTTP {st}: {body[:200]}")
except Exception as e:  # noqa: BLE001
    print(f"GraphQL podTerminate {type(e).__name__}: {e}")
sys.exit(1)
PY
  rc=$?
  [ "$rc" -eq 0 ] && return 0
  [ "$rc" -eq 3 ] && return 3
  if command -v runpodctl >/dev/null 2>&1; then
    RUNPOD_API_KEY=$(guard_pod_key) runpodctl remove pod "$me" && { echo "TERMINATED $me (runpodctl)"; return 0; }
  fi
  return 1
}

# ---- the durability gate (F7) ----------------------------------------------------------------------------------------------------------------
# A pod whose latest stage outputs are not verifiably on the remote must not terminate itself.  Stages register the attempts they published
# (`guard_require_preserved r2026…`; the sweep's run_row_v2.sh does this after every stage); `guard_terminate_self` then refuses unless
# `research data preserved <those ids>` exits 0 — after one `research data push <ids>` repair attempt.  GUARD_FORCE_TERMINATE=1 overrides
# (logged).  The list lives in $GUARD_PRESERVE_FILE (default /root/dm/guard.preserve) so it survives the shell that registered it; the
# GUARD_PRESERVE_RUNS env var (space-separated) is honoured too.  `research` is found as $GUARD_RESEARCH, else `research` on PATH, else
# `python3 -m research` (the shipped tool snapshot is on PYTHONPATH inside a `research run`).
GUARD_PRESERVE_FILE=${GUARD_PRESERVE_FILE:-/root/dm/guard.preserve}

guard_research() {
  if [ -n "${GUARD_RESEARCH:-}" ]; then "$GUARD_RESEARCH" "$@"
  elif command -v research >/dev/null 2>&1; then research "$@"
  else "${GUARD_PY:-python3}" -m research "$@"
  fi
}

guard_require_preserved() {
  # guard_require_preserved RUN_ID|art:...  [more...]  -> appended to the preserve list (deduplicated)
  local id
  mkdir -p "$(dirname "$GUARD_PRESERVE_FILE")" 2>/dev/null || true
  for id in "$@"; do
    [ -n "$id" ] || continue
    grep -qxF "$id" "$GUARD_PRESERVE_FILE" 2>/dev/null || echo "$id" >> "$GUARD_PRESERVE_FILE"
  done
  echo "$(date -u +%FT%TZ) GUARD preserve-required: $*" >> /root/dm/guard.log 2>/dev/null || true
}

guard_preserve_targets() {
  # every registered id, one per line (file + env), deduplicated
  { [ -f "$GUARD_PRESERVE_FILE" ] && cat "$GUARD_PRESERVE_FILE"; for id in ${GUARD_PRESERVE_RUNS:-}; do echo "$id"; done; } 2>/dev/null | awk 'NF && !seen[$0]++'
}

guard_preserved_ok() {
  # 0 when nothing is registered or everything registered is PRESERVED on the remote; 1 (with the list in guard.log) otherwise.
  local targets rc
  targets=$(guard_preserve_targets)
  [ -n "$targets" ] || return 0
  # shellcheck disable=SC2086
  guard_research data preserved $targets >> /root/dm/guard.preserved.log 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date -u +%FT%TZ) GUARD durability gate FAILED rc=$rc for: $(echo $targets | tr '\n' ' ')" >> /root/dm/guard.log
    tail -n 20 /root/dm/guard.preserved.log >> /root/dm/guard.log 2>/dev/null || true
    return 1
  fi
  echo "$(date -u +%FT%TZ) GUARD durability gate ok for: $(echo $targets | tr '\n' ' ')" >> /root/dm/guard.log
  return 0
}

guard_preserve_repair() {
  # one `research data push` of every registered target (the repair the gate names), then re-check
  local targets
  targets=$(guard_preserve_targets)
  [ -n "$targets" ] || return 0
  echo "$(date -u +%FT%TZ) GUARD durability gate: attempting repair (research data push)" >> /root/dm/guard.log
  # shellcheck disable=SC2086
  guard_research data push $targets >> /root/dm/guard.preserved.log 2>&1 || true
  guard_preserved_ok
}

guard_terminate_self() {
  local why=${1:-guard}
  local me; me=$(guard_self_id)
  if [ "${GUARD_FORCE_TERMINATE:-0}" != "1" ]; then
    if ! guard_preserved_ok && ! guard_preserve_repair; then
      echo "$(date -u +%FT%TZ) GUARD REFUSE terminate self ($me): outputs not preserved on the remote ($why); GUARD_FORCE_TERMINATE=1 overrides" \
        | tee -a /root/dm/guard.log ${GUARD_HB:+"$GUARD_HB.log"} >/dev/null 2>&1
      date -u +%FT%TZ > /root/dm/guard.refused 2>/dev/null || true
      return 2
    fi
  else
    echo "$(date -u +%FT%TZ) GUARD durability gate bypassed (GUARD_FORCE_TERMINATE=1)" >> /root/dm/guard.log 2>/dev/null || true
  fi
  echo "$(date -u +%FT%TZ) GUARD terminate self ($me): $why" | tee -a "$GUARD_LOGF" ${GUARD_HB:+"$GUARD_HB.log"} >/dev/null 2>&1
  [ -n "$me" ] || { echo "no RUNPOD_POD_ID; cannot self-terminate" >> "$GUARD_LOGF"; return 1; }
  if [ -n "${GUARD_TERMINATE_CMD:-}" ] || [ -n "$(guard_pod_key)" ] || [ "${GUARD_POD_KEY_ONLY:-0}" = 1 ]; then
    guard_terminate_pod "$me" >> "$GUARD_LOGF" 2>&1
    return $?
  fi
  python3 - "$me" <<'PY' >> "$GUARD_LOGF" 2>&1
import sys; sys.path.insert(0, '/root/dm'); import runpod
me = sys.argv[1]
try:
    runpod._request('DELETE', f'/pods/{me}'); print('TERMINATED', me)
except SystemExit as e:
    print('terminate failed', me, e)
PY
}

# A job process is alive when some process matches the pattern whose command line is NOT the guard itself, its parent shell, or any
# shell that merely mentions the pattern (pgrep -f would match the ssh session that ran `guard_start "<pattern>"`).
guard_job_alive() {
  local pattern=$1 pid cmd
  for pid in $(pgrep -f "$pattern" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    cmd=$( (tr "\0" " " < /proc/"$pid"/cmdline) 2>/dev/null || true)
    case "$cmd" in *guard_start*|*pod_guard*|*guard_job_alive*|"") continue;; esac
    return 0
  done
  return 1
}

guard_start() {
  local pattern=${1:?job process pattern}; local idle_min=${2:-20}
  local who=${GUARD_WHO:-job}
  mkdir -p /root/dm "/workspace/$who/guard" 2>/dev/null || mkdir -p /root/dm
  local me; me=$(guard_self_id)
  export GUARD_HB="/workspace/$who/guard/${me:-nopod}.hb"
  export GUARD_PATTERN="$pattern"
  (
    idle=0
    while true; do
      sleep 60
      date -u +%FT%TZ > "$GUARD_HB" 2>/dev/null || true
      if guard_job_alive "$pattern"; then idle=0; else idle=$((idle+1)); fi
      echo "$(date -u +%FT%TZ) idle=${idle}/${idle_min} pattern=${pattern}" >> /root/dm/guard.log
      if [ "$idle" -ge "$idle_min" ]; then
        if guard_terminate_self "idle ${idle} min, no process matching '${pattern}'"; then exit 0; fi
        idle=$((idle_min > 5 ? idle_min - 5 : 0))     # refused (outputs not preserved) or failed: re-check in 5 min, keep guarding
      fi
    done
  ) &
  export GUARD_PID=$!
  disown "$GUARD_PID" 2>/dev/null || true
  echo "$(date -u +%FT%TZ) GUARD armed pid=$GUARD_PID idle_min=$idle_min pattern='$pattern' hb=$GUARD_HB" | tee -a /root/dm/guard.log
}

guard_finish() {
  local why=${1:-job chain finished}
  sync
  [ -n "${GUARD_PID:-}" ] && kill "$GUARD_PID" 2>/dev/null || true
  guard_terminate_self "$why"
}

# `guard_trap` installs an EXIT trap so that a chain that dies (set -e) still terminates the pod after writing its outputs.
guard_trap() { trap 'rc=$?; guard_finish "chain exit rc=$rc"' EXIT; }

# ---- the per-pod daemon (`research run --on`, started by research.remote.ensure_guard) ----------------------------------------------
# guard_daemon ROOT [IDLE_MIN]: one per pod (flock on <root>/guard/guard.lock where flock exists, else the pidfile; <root>/guard/guard.pid
# names the holder).  Every GUARD_TICK_S (60) `guard_tick` samples activity and decides:
#   busy       a live research runner (<root>/runs/*: non-terminal status.json whose runner or workload pid is alive), a work process,
#              a live verifier, a GPU compute app or GPU utilisation > 0 (the definition research.pods.workproc gives the watcher's
#              probe, passed in as GUARD_WORK_RE / GUARD_NOT_WORK_RE / GUARD_SERVE_RE / GUARD_MARKS_RE)
#   idle       none of that, for less than IDLE_MIN minutes since the last busy tick
#   refuse     idle long enough, but some run dir under <root>/runs has no valid .fetched (written by `research fetch --all` once every
#              file verified; invalid when any file in the run dir changed after the verification began) and no valid .custody
#              (written by a --custody-r2 runner once the attempt and every run file were verified on the store's remote; invalid when
#              a file was added or changed after it) -> logged with the run ids, re-checked every tick
#   terminate  idle long enough and every run fetched -> guard_terminate_self (pod-scoped key only; the F7 preserve gate still applies)
# <root>/guard/state.json (research/pod-guard/v0.1) carries the verdict, last activity, idle minutes and refusal reasons for the laptop.
# Test seams: GUARD_PS (a command printing `ps -eo args=`), GUARD_NVSMI (for nvidia-smi), GUARD_NOW (epoch), GUARD_TERMINATE_CMD.

guard_ps() { if [ -n "${GUARD_PS:-}" ]; then $GUARD_PS; else ps -eo args=; fi; }
guard_nvsmi() {
  if [ -n "${GUARD_NVSMI:-}" ]; then $GUARD_NVSMI "$@"
  elif command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi "$@"
  fi
}

guard_tick() {
  # guard_tick ROOT IDLE_MIN -> prints `VERDICT <busy|idle|refuse|terminate> ...` and rewrites <root>/guard/state.json
  local root=${1:?root} idle_min=${2:?idle minutes} procs
  procs=$(guard_ps 2>/dev/null)
  GUARD_S_WORK=$(printf '%s\n' "$procs" | grep -E "${GUARD_WORK_RE:?}" | grep -vE "${GUARD_NOT_WORK_RE:?}" | grep -c .) \
  GUARD_S_WORK_ARGS=$(printf '%s\n' "$procs" | grep -E "$GUARD_WORK_RE" | grep -vE "$GUARD_NOT_WORK_RE" | head -5 | cut -c1-160) \
  GUARD_S_SERVE=$(printf '%s\n' "$procs" | grep -vE "${GUARD_MARKS_RE:?}" | grep -c -- "${GUARD_SERVE_RE:?}") \
  GUARD_S_GPU=$(guard_nvsmi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
                  | awk 'NF { v = $1 + 0; if (v > u) u = v; n++ } END { if (n) print u + 0; else print "none" }') \
  GUARD_S_APPS=$(guard_nvsmi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .) \
  GUARD_S_POD=$(guard_self_id) \
  "${GUARD_PY:-python3}" - research-pod-guard "$root" "$idle_min" <<'PY'
import json, os, sys, time
root, idle_min = sys.argv[2], float(sys.argv[3])
gdir, runs = os.path.join(root, "guard"), os.path.join(root, "runs")
terminal = set(os.environ.get("GUARD_TERMINAL", "done failed cancelled").split())
now = float(os.environ.get("GUARD_NOW") or time.time())
iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))

def load(p):
    try:
        with open(p) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None

def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, ValueError, TypeError, OverflowError):
        return False

MARKERS = (".fetched", ".fetched.tmp", ".custody", ".custody.tmp", "preserved.json")

def _fetched(d):
    m = load(os.path.join(d, ".fetched"))
    if m is None:
        return "not fetched (no .fetched: run `research fetch --all`)"
    try:
        t0 = float(m["verified_from"])
    except (KeyError, TypeError, ValueError):
        return "unreadable .fetched"
    for base, dirs, files in os.walk(d):
        for f in files:
            if f in (".fetched", ".fetched.tmp"):
                continue
            try:
                if os.lstat(os.path.join(base, f)).st_mtime >= t0:
                    return f"changed after its fetch ({os.path.relpath(os.path.join(base, f), d)})"
            except OSError:
                continue
    return None

def _custody(d):
    # --custody-r2 (research.store.custody.marker_problem): the runner wrote .custody once the attempt and every run file were verified
    # on the store's remote; valid while every regular file is listed in it and none changed at or after verified_from
    m = load(os.path.join(d, ".custody"))
    if m is None:
        return None
    if m.get("schema") != "research/run-custody/v0.1" or not m.get("preserved"):
        return f"custody not on the remote ({str(m.get('error') or '?')[:120]})"
    try:
        t0, listed = float(m["verified_from"]), set(m["files"])
    except (KeyError, TypeError, ValueError):
        return "unreadable .custody"
    for base, dirs, files in os.walk(d):
        for f in files:
            p = os.path.join(base, f)
            rel = os.path.relpath(p, d)
            if rel in MARKERS or os.path.islink(p) or not os.path.isfile(p):
                continue
            if rel not in listed:
                return f"{rel} added after its custody"
            try:
                if os.lstat(p).st_mtime >= t0:
                    return f"{rel} changed after its custody"
            except OSError:
                continue
    return ""

def fetched(d):
    """None when the run dir has custody: a valid .fetched (a verified copy on the launcher) or a valid .custody (on the remote)."""
    why = _fetched(d)
    if why is None:
        return None
    c = _custody(d)
    if c == "":
        return None
    return why if c is None else f"{why}; {c}"

prev = load(os.path.join(gdir, "state.json")) or {}
runners, unfetched, nruns = [], [], 0
for name in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
    d = os.path.join(runs, name)
    if name.startswith(".") or not os.path.isdir(d):
        continue
    nruns += 1
    st, launch = load(os.path.join(d, "status.json")), load(os.path.join(d, "launch.json")) or {}
    if st is None:
        if alive(launch.get("runner_pid")) and not os.path.exists(os.path.join(d, "refused.json")):
            runners.append(name)
    elif st.get("state") not in terminal and (alive(st.get("pid")) or alive(launch.get("runner_pid"))):
        runners.append(name)
    why = fetched(d)
    if why:
        unfetched.append({"run": name, "why": why})

env = os.environ
gpu = env.get("GUARD_S_GPU", "none")
gpu_util = None if gpu in ("", "none") else float(gpu)
act = {"runners": runners, "work": int(env.get("GUARD_S_WORK") or 0), "serve": int(env.get("GUARD_S_SERVE") or 0),
       "gpu_util": gpu_util, "gpu_apps": int(env.get("GUARD_S_APPS") or 0),
       "work_args": [a for a in env.get("GUARD_S_WORK_ARGS", "").splitlines() if a]}
busy = bool(runners or act["work"] or act["serve"] or act["gpu_apps"] or (gpu_util or 0) > 0)
last = now if busy else float(prev.get("last_activity_epoch") or now)
idle_s = max(0.0, now - last)
if busy:
    verdict = "busy"
elif idle_s < idle_min * 60:
    verdict = "idle"
elif unfetched:
    verdict = "refuse"
else:
    verdict = "terminate"
state = {"schema": "research/pod-guard/v0.1", "pod_id": env.get("GUARD_S_POD") or None, "guard_pid": int(env.get("GUARD_DAEMON_PID") or 0) or None,
         "started_utc": prev.get("started_utc") or iso(now), "updated_utc": iso(now), "updated_epoch": now, "idle_min_limit": idle_min,
         "verdict": verdict, "busy": busy, "activity": act, "last_activity_utc": iso(last), "last_activity_epoch": last,
         "idle_min": round(idle_s / 60, 1), "runs": nruns, "unfetched": unfetched,
         "refusal": (f"idle {idle_s / 60:.1f} min >= {idle_min:g} but {len(unfetched)} run(s) not fetched with `research fetch --all`: "
                     + ", ".join(f"{u['run']} ({u['why']})" for u in unfetched)) if verdict == "refuse" else None,
         "refused_since_utc": (prev.get("refused_since_utc") or iso(now)) if verdict == "refuse" else None,
         "terminate": prev.get("terminate")}
os.makedirs(gdir, exist_ok=True)
tmp = os.path.join(gdir, f".state.json.{os.getpid()}")
with open(tmp, "w") as f:
    json.dump(state, f, indent=1)
os.replace(tmp, os.path.join(gdir, "state.json"))
print("VERDICT", verdict, f"idle_min={state['idle_min']}", f"runs={nruns}", "unfetched=" + ",".join(u["run"] for u in unfetched))
PY
}

guard_note_terminate() {
  # guard_note_terminate ROOT RC OUTPUT -> state.json .terminate (the attempt the laptop sees)
  "${GUARD_PY:-python3}" - research-pod-guard "$1" "$2" "$3" <<'PY'
import json, os, sys, time
p = os.path.join(sys.argv[2], "guard", "state.json")
try:
    with open(p) as f:
        st = json.load(f)
except (OSError, ValueError):
    st = {}
st["terminate"] = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rc": int(sys.argv[3]), "ok": sys.argv[3] == "0",
                   "output": sys.argv[4][-400:]}
if sys.argv[3] == "0":
    st["verdict"] = "terminated"
tmp = p + f".{os.getpid()}"
with open(tmp, "w") as f:
    json.dump(st, f, indent=1)
os.replace(tmp, p)
PY
}

guard_is_daemon() {
  # pid $1 is a guard daemon (not a recycled pid)
  local cmd
  cmd=$( (tr "\0" " " < /proc/"$1"/cmdline) 2>/dev/null || ps -o args= -p "$1" 2>/dev/null)
  case "$cmd" in *"pod_guard.sh daemon"*) return 0;; esac
  return 1
}

guard_claim() {
  # single instance: flock (released when the daemon dies) where available, else a noclobber pidfile with a stale-holder check
  local gdir=$1 old
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$gdir/guard.lock"
    flock -n 9 || return 1
    echo $$ > "$gdir/guard.pid"
    return 0
  fi
  if ( set -o noclobber; echo $$ > "$gdir/guard.pid" ) 2>/dev/null; then return 0; fi
  old=$(cat "$gdir/guard.pid" 2>/dev/null)
  if [ -n "$old" ] && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null && guard_is_daemon "$old"; then return 1; fi
  rm -f "$gdir/guard.pid"
  ( set -o noclobber; echo $$ > "$gdir/guard.pid" ) 2>/dev/null
}

guard_daemon() {
  local root=${1:?root} idle_min=${2:-${GUARD_IDLE_MIN:-90}} gdir out verdict key rc last_key=""
  gdir="$root/guard"
  mkdir -p "$gdir"
  GUARD_LOGF="$gdir/guard.log"
  export GUARD_LOGF GUARD_POD_KEY_ONLY=1 GUARD_DAEMON_PID=$$
  if ! guard_claim "$gdir"; then
    echo "$(date -u +%FT%TZ) GUARD daemon $$: another guard holds $gdir (pid $(cat "$gdir/guard.pid" 2>/dev/null)); exiting" >> "$GUARD_LOGF"
    return 0
  fi
  echo "$(date -u +%FT%TZ) GUARD daemon armed pid=$$ pod=$(guard_self_id) idle_min=$idle_min root=$root" >> "$GUARD_LOGF"
  rm -f "$gdir/state.json"               # a new daemon starts its idle clock now (a launch is activity)
  while :; do
    out=$(guard_tick "$root" "$idle_min" 2>&1)
    verdict=$(printf '%s\n' "$out" | sed -n 's/^VERDICT \([a-z]*\).*/\1/p' | tail -1)
    key=$(printf '%s\n' "$out" | grep '^VERDICT' | sed 's/ idle_min=[^ ]*//')
    if [ -z "$verdict" ]; then
      echo "$(date -u +%FT%TZ) GUARD tick failed: $(printf '%s' "$out" | tail -3)" >> "$GUARD_LOGF"
    elif [ "$key" != "$last_key" ]; then       # every change of verdict or of the refusing run ids
      echo "$(date -u +%FT%TZ) $out" >> "$GUARD_LOGF"
    fi
    last_key=$key
    if [ "$verdict" = terminate ]; then
      out=$(guard_terminate_self "idle >= ${idle_min} min, every run fetched" 2>&1); rc=$?
      guard_note_terminate "$root" "$rc" "$out$(tail -3 "$GUARD_LOGF" 2>/dev/null)"
      [ "$rc" -eq 0 ] && { echo "$(date -u +%FT%TZ) GUARD daemon $$: termination accepted; exiting" >> "$GUARD_LOGF"; return 0; }
      sleep "${GUARD_RETRY_S:-300}" 9>&-      # refused by the F7 gate or the API failed: keep guarding, do not hammer the API
    fi
    sleep "${GUARD_TICK_S:-60}" 9>&-
  done
}

if [ "${1:-}" = daemon ]; then
  shift
  guard_daemon "$@"
fi
