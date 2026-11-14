set -u
# The check slot exports CHECK_SLOT_BOOK_GB, which a tree without 70cd2660ede does not declare.
git grep -q CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py || unset CHECK_SLOT_BOOK_GB
MAIN=56f6c267563373883c093381ec5bf02a0ec80880
ONE="tests/test_public_boundary.py::test_this_trees_export_rewrites_every_manifest_it_copies"
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$MAIN" HEAD && echo "holds main $MAIN" || { echo "DOES NOT HOLD main $MAIN"; exit 8; }
git status --short | head -5
uv sync -q --all-packages --extra torch-cpu || { echo "SYNC_FAILED"; exit 9; }
uv run pytest "$ONE" -q -p no:cacheprovider > "$OUT/one-test.log" 2>&1; echo "ONE_TEST_EXIT $?" >> "$OUT/one-test.log"
tail -4 "$OUT/one-test.log"
( uv run tools/verity/check/suites.py repository --fresh > "$OUT/repository.log" 2>&1; echo "REPOSITORY_EXIT $?" >> "$OUT/repository.log"
  uv run tools/verity/check/suites.py --changed "$MAIN" --quick > "$OUT/suites.log" 2>&1; echo "SUITES_EXIT $?" >> "$OUT/suites.log" ) &
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
tail -12 "$OUT/repository.log"
tail -40 "$OUT/suites.log"
git status --short > "$OUT/tree.txt" 2>&1
git diff > "$OUT/tree.patch" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
O=$(tail -1 "$OUT/one-test.log" | sed 's/ONE_TEST_EXIT //')
R=$(tail -1 "$OUT/repository.log" | sed 's/REPOSITORY_EXIT //')
S=$(tail -1 "$OUT/suites.log" | sed 's/SUITES_EXIT //')
echo "ONE_TEST_EXIT $O REPOSITORY_EXIT $R SUITES_EXIT $S LEAN_EXIT $LEAN"
[ "$O" = 0 ] && [ "$R" = 0 ] && [ "$S" = 0 ] && [ "$LEAN" = 0 ]
