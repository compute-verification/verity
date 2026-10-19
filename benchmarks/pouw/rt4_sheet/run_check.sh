#!/usr/bin/env bash
# RT4-SHEET job 4 on the tree it runs in: the exporter's --check, `machine-model headline --json` and sheet_check.py.
# On this branch's tree, or on another PR's head with the comparator sent:
#
#   research run --on vy-nebius-cpu-3 --project verity --source <tree> --cwd source --mem-gb 8 --timeout 1800 \
#     [--send benchmarks/pouw/rt4_sheet/sheet_check.py] -- bash benchmarks/pouw/rt4_sheet/run_check.sh
#
# Exit: the exporter's --check status (0: the committed Lean is what it writes), else 4 when sheet_check disagrees.
set -uo pipefail
OUT=${OUT:?run under research run}
CMP=benchmarks/pouw/rt4_sheet/sheet_check.py
[ -f "$CMP" ] || CMP="${RESEARCH_RUN_DIR:?}/inputs/sheet_check.py"
git rev-parse HEAD >"$OUT/tree.sha"
uv sync -q --all-packages --extra torch-cpu || exit 2
uv run --no-sync python benchmarks/pouw/lean_headline.py --check >"$OUT/lean_headline_check.txt" 2>&1
lean=$?
uv run --no-sync machine-model headline --json >"$OUT/headline.json" 2>"$OUT/headline.err"
uv run --no-sync python "$CMP" --tree . >"$OUT/sheet_check.json" 2>"$OUT/sheet_check.err"
cmp=$?
printf 'lean_headline_check=%s sheet_check=%s\n' "$lean" "$cmp" | tee "$OUT/rc.txt"
[ "$lean" -ne 0 ] && exit "$lean"
[ "$cmp" -ne 0 ] && exit 4
exit 0
