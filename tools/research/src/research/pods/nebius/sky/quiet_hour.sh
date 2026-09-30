#!/bin/sh
# quiet_hour.sh hold | release | status: the daily quiet hour on vy-nebius-1 (12:30-13:30Z, systemd timers from cluster_up.sh).
# hold stops circuits admitting new workloads (stopPolicy: Hold): no new config run, so benches re-measure without a Build
# starting beside them. Workloads admitted before the hold keep running (a config run can't be checkpointed), and direct
# `research run --on` Builds are outside Kueue: lanes keep the hour free themselves.
set -eu
K=${VY_KUBECTL:-k3s kubectl}
case ${1:-status} in
  hold) policy=Hold ;;
  release) policy=None ;;
  status) $K get clusterqueue circuits -o jsonpath='{.spec.stopPolicy}{"\n"}'; exit 0 ;;
  *) echo "usage: quiet_hour.sh hold | release | status" >&2; exit 2 ;;
esac
$K patch clusterqueue circuits --type=merge -p "{\"spec\":{\"stopPolicy\":\"$policy\"}}"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) circuits stopPolicy=$policy"
