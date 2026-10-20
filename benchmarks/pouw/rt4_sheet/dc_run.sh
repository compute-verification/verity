#!/usr/bin/env bash
# dc_price.py on a tree whose DC statement is under test (DC-CENSUS's head of #2699), with fold_bulk_sass.sh's SASS as
# inputs; also that tree's own `lean_headline_dc.py --check`. From the tree under test:
#
#   research run --on vy-nebius-cpu-3 --project verity --campaign eps1 --source <that tree> --cwd source --mem-gb 16 \
#     --timeout 3600 --send <rt4 tree>/benchmarks/pouw/rt4_sheet/dc_price.py \
#     --send <rt4 tree>/benchmarks/pouw/rt4_sheet/dc_run.sh --send fold_bulk_sm_90a.sass \
#     --send fold_bulk_sm_100a.sass -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/dc_run.sh"'
#
# Exit: the exporter's --check rc, else 4 when a route prices under φ* on a sheet.
set -uo pipefail
OUT=${OUT:?run under research run}
IN=${RESEARCH_RUN_DIR:?}/inputs
git rev-parse HEAD >"$OUT/tree.sha"
uv sync -q --all-packages --extra torch-cpu
uv run --no-sync python benchmarks/pouw/lean_headline_dc.py --check >"$OUT/lean_headline_dc_check.txt" 2>&1
lean=$?
uv run --no-sync python benchmarks/pouw/lean_headline_dc.py --numbers >"$OUT/lean_headline_dc_numbers.txt" 2>&1
uv run --no-sync python "$IN/dc_price.py" --tree . "$IN/fold_bulk_sm_90a.sass" "$IN/fold_bulk_sm_100a.sass" \
  --json "$OUT/dc_price.json" 2>&1 | tee "$OUT/dc_price.txt"
price=${PIPESTATUS[0]}
echo "lean_headline_dc --check rc $lean; dc_price rc $price" | tee "$OUT/rc.txt"
[ "$lean" = 0 ] || exit "$lean"
[ "$price" = 0 ] || exit 4
