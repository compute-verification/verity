#!/bin/bash
# quiet_cell.sh BUDGET_MIN UNTIL AVOID[,AVOID...] -- CELL_ARGS...: one P2 sweep cell on vy-nebius-2's quiet slice (vy-pous-quiet, cores
# 124-127, GPU 7) under a timed GPU-7 lease, which serializes it with every other timed run there. It waits on the node until a
# BUDGET_MIN lease from now clears every held window AVOID (START/END, UTC), takes the lease, and runs `sweep.py cell` with the
# same guard; a cell whose lease came too late refuses (75) and the loop waits again, until a lease would end past UNTIL (UTC).
set -u
budget=$1 until=$2 avoid=$3
shift 3
[ "${1:-}" = -- ] && shift
here=$(cd "$(dirname "$0")" && pwd)
windows=()
for w in ${avoid//,/ }; do windows+=(--avoid "$w"); done
while :; do
  python3 "$here/sweep.py" clear "${windows[@]}" --budget-min "$budget" --until "$until" || exit 75
  # node_ops' Quiet stops this user's busy process groups outside the lease's process tree while a timed lease is held, and
  # vy-pous-quiet's sudo (root's) cuts that tree; a group under GNU timeout is exempt, so the cell runs as one.
  gpu-lease 1 --on 7 --wait --max-min "$budget" --timed -- \
    vy-pous-quiet timeout -k 20 $((budget * 60 - 30)) python3 "$here/sweep.py" cell "$@" --quiet-slice "${windows[@]}" \
    --budget-min "$budget"
  rc=$?
  [ "$rc" = 75 ] || exit "$rc"
done
