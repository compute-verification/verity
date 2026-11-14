set -u
# The carried #9636 at its head: the suites its change touches against the main it holds (the quick tier), the ratchet
# gate's verdict, the Markdown caps gate and the retired-terms lint by name.
git grep -q CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py || unset CHECK_SLOT_BOOK_GB
BASE=@BASE@
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$BASE" HEAD && echo "holds main $BASE" || { echo "DOES NOT HOLD main $BASE"; exit 8; }
git diff --stat "$BASE" HEAD | tail -1
uv sync -q --all-packages --extra torch-cpu || { echo "SYNC_FAILED"; exit 9; }
python3 tools/verity/check/gates/markdown_caps.py "$BASE" HEAD > "$OUT/caps.log" 2>&1; CAPS=$?
echo "CAPS_EXIT $CAPS" >> "$OUT/caps.log"; tail -4 "$OUT/caps.log" | cut -c1-400
python3 adapters/vllm/tools/ratchet_gate.py "$BASE" HEAD > "$OUT/gate.log" 2>&1; GATE=$?
echo "GATE_EXIT $GATE" >> "$OUT/gate.log"; tail -4 "$OUT/gate.log" | cut -c1-400
uv run python -m pytest tests/lint/test_repo_retired_terms.py -q -p no:cacheprovider > "$OUT/lint.log" 2>&1; LINT=$?
echo "LINT_EXIT $LINT" >> "$OUT/lint.log"; tail -3 "$OUT/lint.log" | cut -c1-300
uv run tools/verity/check/suites.py --changed "$BASE" --quick > "$OUT/suites.log" 2>&1; SUITES=$?
echo "SUITES_EXIT $SUITES" >> "$OUT/suites.log"
tail -40 "$OUT/suites.log" | cut -c1-300
git status --short > "$OUT/tree.txt" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
echo "CAPS_EXIT $CAPS GATE_EXIT $GATE LINT_EXIT $LINT SUITES_EXIT $SUITES"
[ "$CAPS" = 0 ] && [ "$GATE" = 0 ] && [ "$LINT" = 0 ] && [ "$SUITES" = 0 ]
