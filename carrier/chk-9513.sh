set -u
MAIN=56f6c267563373883c093381ec5bf02a0ec80880
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$MAIN" HEAD && echo "holds main $MAIN" || { echo "DOES NOT HOLD main $MAIN"; exit 8; }
git status --short | head -5
uv sync -q --all-packages --extra torch-cpu || { echo "SYNC_FAILED"; exit 9; }
( uv run tools/verity/check/suites.py --changed "$MAIN" --quick > "$OUT/suites.log" 2>&1; echo "SUITES_EXIT $?" >> "$OUT/suites.log" ) &
SUITES=$!
python3 tools/verity/lean/lean_changed.py --records security/Proofs > "$OUT/lean.log" 2>&1; LEAN=$?
echo "LEAN_EXIT $LEAN" >> "$OUT/lean.log"
tail -25 "$OUT/lean.log"
RESEARCH_RUN_DIR="${RESEARCH_RUN_DIR:-}"
if [ -n "$RESEARCH_RUN_DIR" ] && [ -d "$RESEARCH_RUN_DIR/lean-audit" ]; then
  mkdir -p "$OUT/lean-audit"
  for f in result.json record allowed-record; do
    [ -e "$RESEARCH_RUN_DIR/lean-audit/$f" ] && cp -a "$RESEARCH_RUN_DIR/lean-audit/$f" "$OUT/lean-audit/"
  done
  ( cd "$RESEARCH_RUN_DIR/lean-audit" && find . -maxdepth 5 \( -name lean-audit.json -o -name report.json -o -name result.json \) | sort | head -120 ) > "$OUT/lean-audit/files.txt" 2>&1
fi
wait $SUITES
tail -40 "$OUT/suites.log"
git status --short > "$OUT/tree.txt" 2>&1
git diff > "$OUT/tree.patch" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
S=$(tail -1 "$OUT/suites.log" | sed 's/SUITES_EXIT //')
echo "SUITES_EXIT $S LEAN_EXIT $LEAN"
[ "$S" = 0 ] && [ "$LEAN" = 0 ]
