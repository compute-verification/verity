#!/usr/bin/env bash
# The serving cost's check on vy-dc-h200-2 once the stock control CTL (stock.sh's) has ended: check/check_stock.sh at the
# run's tree, odserve_check.sh of the served run RID against CTL's aggregate, its lines in the run's out/odserve.txt.
# Waited for up to 20 minutes; its launch in LOGS/check-RID.txt, then its lines (runlog.sh h200-2 ID all). The cost's
# label: the certifier's pacing plus H1's commits, against a same-engine stock control.
#   [DRY=1] bash check.sh RID CTL
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
RID=${1:?the served run id} CTL=${2:?the stock control run id}
for r in "$RID" "$CTL"; do [[ $r =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || { echo "REFUSED: $r is not a store run id"; exit 3; }; done
tree_ok || exit 3
cmd=(research run --on "$SERVED" --project verity --source "$TREE" --cwd clone --timeout 20m --wait --declared-output 'out/*'
  --env "RUN=$RID" --env "CTL=$CTL" --send "$H/check/check_stock.sh" --send "$H/check/odserve_check.sh"
  --send "$H/check/odserve_check.py" --send "$H/check/odserve_cost.py" -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/check_stock.sh"')
[ -z "${DRY:-}" ] || { printf '%q ' "${cmd[@]}"; echo; exit 0; }
(cd "$TREE" && "${cmd[@]}") > "$LOGS/check-$RID.txt" 2>&1
rc=$?
cid=$(grep -o -E 'run r[0-9]{8}-[0-9]{6}-[0-9a-f]{4} launched' "$LOGS/check-$RID.txt" | head -1 | cut -d' ' -f2)
echo "CHECK ${cid:-not launched} of $RID against $CTL: exit $rc"
[ -n "$cid" ] && bash "$H/runlog.sh" h200-2 "$cid" all
