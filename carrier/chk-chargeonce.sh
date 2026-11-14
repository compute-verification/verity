set -u
# The ChargeOnce.lean change as a pull request's head, one commit over main (@BASE@: the commit of main it is cut from,
# written in by launch-auto.sh): the retired-terms lint by name, the ratchet gate's verdict on the change, the suites the
# change touches (the pre-mark run), then the Lean: the try of the file with the sibling check, the build of its module
# and of the module that imports it, and the Lean check of its package's records (the fast path, no --update).
git grep -q CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py || unset CHECK_SLOT_BOOK_GB
BASE=@BASE@
F=security/Proofs/Flock/Claims/ChargeOnce.lean
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$BASE" HEAD && echo "holds main $BASE" || { echo "DOES NOT HOLD main $BASE"; exit 8; }
N=$(git rev-list --count "$BASE"..HEAD)
[ "$N" = 1 ] && echo "one commit over main" || { echo "NOT ONE COMMIT OVER main: $N"; exit 8; }
git diff --stat "$BASE" HEAD | tail -3
git status --short | head -5
uv sync -q --all-packages --extra torch-cpu || { echo "SYNC_FAILED"; exit 9; }
uv run python -m pytest tests/lint/test_repo_retired_terms.py -q -p no:cacheprovider > "$OUT/lint.log" 2>&1; LINT=$?
echo "LINT_EXIT $LINT" >> "$OUT/lint.log"
tail -3 "$OUT/lint.log" | cut -c1-300
python3 adapters/vllm/tools/ratchet_gate.py "$BASE" HEAD > "$OUT/gate.log" 2>&1; GATE=$?
echo "GATE_EXIT $GATE" >> "$OUT/gate.log"
tail -4 "$OUT/gate.log" | cut -c1-400
uv run tools/verity/check/suites.py --changed "$BASE" --quick > "$OUT/suites.log" 2>&1; SUITES=$?
echo "SUITES_EXIT $SUITES" >> "$OUT/suites.log"
tail -30 "$OUT/suites.log" | cut -c1-300
python3 tools/verity/lean/lean_changed.py --try "$F" --siblings > "$OUT/try.log" 2>&1; TRY=$?
echo "TRY_EXIT $TRY" >> "$OUT/try.log"
tail -8 "$OUT/try.log" | cut -c1-600
python3 tools/verity/lean/lean_changed.py --build "$F" security/Proofs/Flock/Claims.lean > "$OUT/build.log" 2>&1; BUILD=$?
echo "BUILD_EXIT $BUILD" >> "$OUT/build.log"
tail -3 "$OUT/build.log" | cut -c1-600
python3 tools/verity/lean/lean_changed.py --records security/Proofs > "$OUT/lean.log" 2>&1; LEAN=$?
echo "LEAN_EXIT $LEAN" >> "$OUT/lean.log"
tail -25 "$OUT/lean.log" | cut -c1-400
RESEARCH_RUN_DIR="${RESEARCH_RUN_DIR:-}"
if [ -n "$RESEARCH_RUN_DIR" ] && [ -d "$RESEARCH_RUN_DIR/lean-audit" ]; then
  mkdir -p "$OUT/lean-audit"
  for f in result.json record allowed-record; do
    [ -e "$RESEARCH_RUN_DIR/lean-audit/$f" ] && cp -a "$RESEARCH_RUN_DIR/lean-audit/$f" "$OUT/lean-audit/"
  done
  ( cd "$RESEARCH_RUN_DIR/lean-audit" && find . -maxdepth 5 \( -name lean-audit.json -o -name report.json -o -name result.json \) | sort | head -120 ) > "$OUT/lean-audit/files.txt" 2>&1
fi
git status --short > "$OUT/tree.txt" 2>&1
git diff > "$OUT/tree.patch" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
echo "LINT_EXIT $LINT GATE_EXIT $GATE SUITES_EXIT $SUITES TRY_EXIT $TRY BUILD_EXIT $BUILD LEAN_EXIT $LEAN"
[ "$LINT" = 0 ] && [ "$GATE" = 0 ] && [ "$SUITES" = 0 ] && [ "$TRY" = 0 ] && [ "$BUILD" = 0 ] && [ "$LEAN" = 0 ]
