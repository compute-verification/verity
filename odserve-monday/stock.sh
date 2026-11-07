#!/usr/bin/env bash
# The serving cost's same-engine stock control on vy-dc-h200-2, after the served run RID and its send (it leases the GPU:
# never beside a serve, and not beside the send's copy, which it would slow): check/stock_control.sh at the run's tree,
# at RID's engine and load without the network certifier: remote's settings with the served run's WINDOW_STEPS, which
# the job reads on the node from RID's private deferred.env (so the count never leaves the node), PRED the served run's
# predicted dir and the traced row; RUN=later keeps its aggregate for check.sh. Prints the control's id.
#   [DRY=1] bash stock.sh RID
set -euo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh"
RID=${1:?the served run id}
[[ $RID =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || { echo "REFUSED: $RID is not a store run id"; exit 3; }
tree_ok
# shellcheck disable=SC2016
job='WINDOW_STEPS=$(set +u; . "$SERVED_DIR/private/deferred/deferred.env" > /dev/null 2>&1; echo "${WINDOW_STEPS:-}")
[ -n "$WINDOW_STEPS" ] || { echo "PREFLIGHT: the served run'"'"'s deferred.env names no WINDOW_STEPS"; exit 3; }
export WINDOW_STEPS; exec bash "$RESEARCH_RUN_DIR/inputs/stock_control.sh"'
cmd=(research run --on "$SERVED" --project verity --source "$TREE" --cwd clone --timeout 60m --declared-output 'out/*'
  --send "$H/check/stock_control.sh" --send "$H/check/odserve_check.sh" --send "$H/check/odserve_check.py"
  --send "$H/check/odserve_cost.py" --env RUN=later --env "SERVED_DIR=$SERVE_ROOT/$RID" --env "PRED=$SERVE_ROOT/$RID/predicted"
  --env "ROW=$ROW" -- bash -c "$job")
echo "TREE $(git -C "$TREE" rev-parse --short=11 HEAD), the stock control of $RID"
[ -z "${DRY:-}" ] || { printf '%q ' "${cmd[@]}"; echo; exit 0; }
cd "$TREE"
"${cmd[@]}" 2>&1 | tee "$LOGS/stock-$RID.txt"
