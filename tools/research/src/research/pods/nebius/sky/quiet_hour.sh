#!/bin/sh
# quiet_hour.sh hold | release | status: the daily quiet hour on vy-nebius-1 (12:30-13:30Z, systemd timers from cluster_up.sh).
# hold stops the vLLM deployments' queues (VY_QUIET_QUEUES: deployments-cpu deployments-gpu circuits) admitting new workloads (stopPolicy: Hold): no new config run, so benches re-measure without a Build
# starting beside them. Workloads admitted before the hold keep running (a config run can't be checkpointed), and direct
# `research run --on` Builds are outside Kueue: lanes keep the hour free themselves.
set -eu
K=${VY_KUBECTL:-k3s kubectl}
QUEUES=${VY_QUIET_QUEUES:-deployments-cpu deployments-gpu circuits}
case ${1:-status} in
  hold) policy=Hold ;;
  release) policy=None ;;
  status) for q in $QUEUES; do echo "$q $($K get clusterqueue "$q" -o jsonpath='{.spec.stopPolicy}')"; done; exit 0 ;;
  *) echo "usage: quiet_hour.sh hold | release | status" >&2; exit 2 ;;
esac
for q in $QUEUES; do
  $K patch clusterqueue "$q" --type=merge -p "{\"spec\":{\"stopPolicy\":\"$policy\"}}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $q stopPolicy=$policy"
done
