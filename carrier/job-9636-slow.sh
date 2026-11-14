set -u
# The carried #9636 at its head: the SPOT verifier's two suites whole, slow tests included (no --quick): the verifier's
# own and its Lean's.
git grep -q CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py || unset CHECK_SLOT_BOOK_GB
echo "head $(git rev-parse HEAD)"
uv sync -q --all-packages --extra torch-cpu || { echo "SYNC_FAILED"; exit 9; }
uv run tools/verity/check/suites.py verity/protocols/proofs/spot/verifier verity/protocols/proofs/spot/verifier/lean > "$OUT/slow.log" 2>&1; SLOW=$?
echo "SLOW_EXIT $SLOW" >> "$OUT/slow.log"
tail -40 "$OUT/slow.log" | cut -c1-300
git status --short > "$OUT/tree.txt" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
echo "SLOW_EXIT $SLOW"
[ "$SLOW" = 0 ]
