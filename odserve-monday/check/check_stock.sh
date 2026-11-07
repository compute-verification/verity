#!/usr/bin/env bash
# odserve_check.sh of the served run RUN against a stock control's aggregate on this node (stock_control.sh with RUN=later):
#   research run --on vy-dc-h200-2 --project verity --source TREE --cwd clone --timeout 20m --send check_stock.sh \
#     --send odserve_check.sh --send odserve_check.py --send odserve_cost.py --env RUN=RID --env CTL=CONTROL_RID -- \
#     bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/check_stock.sh"'
set -euo pipefail
IN=$RESEARCH_RUN_DIR/inputs
f=${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}/${CTL:?the stock control run}/private/stock-overhead.json
[ -s "$f" ] || { echo "PREFLIGHT: no stock aggregate $f"; exit 3; }
RUN=${RUN:?the served run} STOCK=$(realpath --relative-to="$IN" "$f") exec bash "$IN/odserve_check.sh"
