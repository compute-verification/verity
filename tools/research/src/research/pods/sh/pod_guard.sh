#!/usr/bin/env bash
# Pod-side guard (integrator, 2026-09-13, D55) — sourced ON THE POD by every job launcher.  Nothing here depends on the laptop or on
# the agent that launched the job being alive.  Requires /root/.runpod/config.toml (the RunPod API key) and /root/dm/runpod.py on the pod
# (both are copied by the launcher; deadman.sh does the same for the controller pod) and the RUNPOD_POD_ID env var RunPod sets in every pod.
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
#   * heartbeat      — /workspace/<who>/guard/<pod>.hb is touched every minute so the state can be read from the volume by anyone.
# The daemon-side per-pod age ceiling (budget_cap.py --pod-max-hours) and the cap/rate/deadline on the controller pod remain the outer layers.

guard_self_id() {
  # RunPod sets RUNPOD_POD_ID for the container's main process; ssh sessions do not inherit it -> read /proc/1/environ, then /root/dm/POD_ID.
  if [ -n "${RUNPOD_POD_ID:-}" ]; then echo "$RUNPOD_POD_ID"; return; fi
  tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' | head -1 | grep . && return
  cat /root/dm/POD_ID 2>/dev/null
}

guard_terminate_self() {
  local why=${1:-guard}
  local me; me=$(guard_self_id)
  echo "$(date -u +%FT%TZ) GUARD terminate self ($me): $why" | tee -a /root/dm/guard.log ${GUARD_HB:+"$GUARD_HB.log"} >/dev/null 2>&1
  [ -n "$me" ] || { echo "no RUNPOD_POD_ID; cannot self-terminate" >> /root/dm/guard.log; return 1; }
  python3 - "$me" <<'PY' >> /root/dm/guard.log 2>&1
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
      if [ "$idle" -ge "$idle_min" ]; then guard_terminate_self "idle ${idle} min, no process matching '${pattern}'"; exit 0; fi
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
