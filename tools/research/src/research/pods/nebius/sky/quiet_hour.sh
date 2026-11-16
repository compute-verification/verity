#!/bin/sh
# quiet_hour.sh hold | release | status: the daily quiet hour on vy-nebius-1 (12:30-13:30Z, systemd timers from cluster_up.sh).
# hold stops the vLLM deployments' queues (VY_QUIET_QUEUES: deployments-cpu deployments-gpu circuits) admitting new workloads (stopPolicy: Hold): no new config run, so benches re-measure without a Build
# starting beside them. Workloads admitted before the hold keep running (a config run can't be checkpointed), and direct
# `research run --on` Builds are outside Kueue: lanes keep the hour free themselves.
# Other writers hold these queues too (vy-disk-guard, a lane by hand). hold takes only a queue nobody holds and lists it in
# VY_QUIET_STATE; release frees only the queues listed there, and none that the disk guard lists as its own (VY_DISK_STATE), so
# a hold set by anyone else outlasts the hour.
set -eu
K=${VY_KUBECTL:-k3s kubectl}
QUEUES=${VY_QUIET_QUEUES:-deployments-cpu deployments-gpu circuits}
STATE=${VY_QUIET_STATE:-/var/lib/vy-quiet-hour/held}
DISK_STATE=${VY_DISK_STATE:-/var/lib/vy-disk-guard/held}

policy() { $K get clusterqueue "$1" -o jsonpath='{.spec.stopPolicy}'; }
set_policy() { $K patch clusterqueue "$1" --type=merge -p "{\"spec\":{\"stopPolicy\":\"$2\"}}" > /dev/null; }
held() { [ ! -f "$STATE" ] || tr '\n' ' ' < "$STATE" | sed 's/ *$//'; }
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

case ${1:-status} in
  status)
    echo "held by the quiet hour: $(held)"
    for q in $QUEUES; do echo "$q $(policy "$q")"; done
    exit 0 ;;
  hold)
    mkdir -p "$(dirname "$STATE")"
    touch "$STATE"
    for q in $QUEUES; do
      p=$(policy "$q")
      if grep -qx "$q" "$STATE"; then
        [ "$p" = Hold ] || set_policy "$q" Hold
        say "$q stopPolicy=Hold"
      elif [ "$p" = Hold ] || [ "$p" = HoldAndDrain ]; then
        say "$q stopPolicy=$p, held by someone else: left alone"
      else
        set_policy "$q" Hold
        echo "$q" >> "$STATE"
        say "$q stopPolicy=Hold"
      fi
    done ;;
  release)
    for q in $(held); do
      p=$(policy "$q")
      if grep -qx "$q" "$DISK_STATE" 2> /dev/null; then
        say "$q stopPolicy=$p, held by the disk guard: left alone"
      elif [ "$p" = Hold ]; then
        set_policy "$q" None
        say "$q stopPolicy=None"
      else
        say "$q stopPolicy=${p:-None}, changed since the hold: left alone"
      fi
    done
    [ ! -f "$STATE" ] || : > "$STATE" ;;
  *) echo "usage: quiet_hour.sh hold | release | status" >&2; exit 2 ;;
esac
