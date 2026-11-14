set -u
# The ChargeOnce.lean change (its three theorems in a namespace of their own, its retired-terms entry 5 -> 2), on a scratch
# branch over #9513's head: a try of the file with the sibling check, the build of its module and of the module that
# imports it, the retired-terms lint by name, and the ratchet gate's verdict on the change.
git grep -q CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py || unset CHECK_SLOT_BOOK_GB
BASE=${BASE_SHA:-a2643447fa06a78b6467633fbbf6ebfd2f1c6d42}
F=security/Proofs/Flock/Claims/ChargeOnce.lean
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$BASE" HEAD && echo "holds $BASE" || { echo "DOES NOT HOLD $BASE"; exit 8; }
git diff --stat "$BASE" HEAD | tail -3
uv sync -q --all-packages --extra torch-cpu || { echo "SYNC_FAILED"; exit 9; }
python3 tools/verity/lean/lean_changed.py --try "$F" --siblings > "$OUT/try.log" 2>&1; TRY=$?
echo "TRY_EXIT $TRY" >> "$OUT/try.log"
tail -8 "$OUT/try.log" | cut -c1-600
python3 tools/verity/lean/lean_changed.py --build "$F" security/Proofs/Flock/Claims.lean > "$OUT/build.log" 2>&1; BUILD=$?
echo "BUILD_EXIT $BUILD" >> "$OUT/build.log"
tail -3 "$OUT/build.log" | cut -c1-600
uv run python -m pytest tests/lint/test_repo_retired_terms.py -q -p no:cacheprovider > "$OUT/lint.log" 2>&1; LINT=$?
echo "LINT_EXIT $LINT" >> "$OUT/lint.log"
tail -3 "$OUT/lint.log" | cut -c1-300
python3 adapters/vllm/tools/ratchet_gate.py "$BASE" HEAD > "$OUT/gate.log" 2>&1; GATE=$?
echo "GATE_EXIT $GATE" >> "$OUT/gate.log"
tail -4 "$OUT/gate.log" | cut -c1-400
git status --short > "$OUT/tree.txt" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
echo "TRY_EXIT $TRY BUILD_EXIT $BUILD LINT_EXIT $LINT GATE_EXIT $GATE"
[ "$TRY" = 0 ] && [ "$BUILD" = 0 ] && [ "$LINT" = 0 ] && [ "$GATE" = 0 ]
