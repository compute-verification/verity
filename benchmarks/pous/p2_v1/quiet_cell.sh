#!/bin/bash
# quiet_cell.sh BUDGET_MIN UNTIL AVOID[,AVOID...] -- CELL_ARGS...: one P2 sweep cell on vy-nebius-2's quiet slice (vy-pous-quiet, cores
# 124-127, GPU 7) under a timed GPU-7 lease, which serializes it with every other timed run there. It waits on the node until a
# BUDGET_MIN lease from now clears every held window AVOID (START/END, UTC), takes the lease, and runs `sweep.py cell` with the
# same guard; a cell whose lease came too late refuses (75) and the loop waits again, until UNTIL (UTC) passes.
set -u
budget=$1 until=$2 avoid=$3
shift 3
[ "${1:-}" = -- ] && shift
here=$(cd "$(dirname "$0")" && pwd)
windows=()
for w in ${avoid//,/ }; do windows+=(--avoid "$w"); done
while :; do
  python3 "$here/sweep.py" clear "${windows[@]}" --budget-min "$budget" --until "$until" || exit 75
  gpu-lease 1 --on 7 --wait --max-min "$budget" --timed -- \
    vy-pous-quiet python3 "$here/sweep.py" cell "$@" --quiet-slice "${windows[@]}" --budget-min "$budget"
  rc=$?
  [ "$rc" = 75 ] || exit "$rc"
done
