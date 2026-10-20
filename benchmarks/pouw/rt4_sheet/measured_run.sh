#!/usr/bin/env bash
# measured_price.py on a tree whose sm_120 statements are under test (#2791's head), with fold_bulk.sh's sm_120a SASS
# (the device run's `sass/fold_bulk.sass`) as an input. From the tree under test:
#
#   research run --on vy-nebius-cpu-3 --project verity --campaign eps1 --source <that tree> --cwd source --mem-gb 8 \
#     --timeout 3600 --send <rt4 tree>/benchmarks/pouw/rt4_sheet/measured_price.py \
#     --send <rt4 tree>/benchmarks/pouw/rt4_sheet/dc_price.py --send <rt4 tree>/benchmarks/pouw/rt4_sheet/measured_run.sh \
#     --send fold_bulk.sass -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/measured_run.sh"'
#
# Exit: 4 when a route prices under φ* on either reading, else measured_price.py's rc.
set -uo pipefail
OUT=${OUT:?run under research run}
IN=${RESEARCH_RUN_DIR:?}/inputs
git rev-parse HEAD >"$OUT/tree.sha"
uv sync -q --all-packages --extra torch-cpu
uv run --no-sync python "$IN/measured_price.py" --tree . "$IN/fold_bulk.sass" --json "$OUT/measured_price.json" 2>&1 |
  tee "$OUT/measured_price.txt"
rc=${PIPESTATUS[0]}
echo "measured_price rc $rc" | tee "$OUT/rc.txt"
[ "$rc" = 1 ] && exit 4
exit "$rc"
