set -u
# #9584's carried head merged with main (a scratch merge, never #9584's branch): Spot's Lean built, Spot's verifier's
# two suites and benchmarks/one_stage whole, and the records' fast path without --update.
git grep -q CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py || unset CHECK_SLOT_BOOK_GB
MAIN=6953e38e55a4283227d880418fbd833f0fa0588b
CARRIED=c9ba2821ff0593a390289bc04ee55d004100b988
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$MAIN" HEAD && echo "holds main $MAIN" || { echo "DOES NOT HOLD main $MAIN"; exit 8; }
git merge-base --is-ancestor "$CARRIED" HEAD && echo "holds the carried head $CARRIED" || { echo "DOES NOT HOLD $CARRIED"; exit 8; }
git status --short | head -5
uv sync -q --all-packages --extra torch-cpu || { echo "SYNC_FAILED"; exit 9; }
FILES=$(git ls-files 'verity/protocols/proofs/spot/verifier/lean/*.lean' 'verity/protocols/proofs/spot/verifier/lean/**/*.lean' | sort -u)
echo "build: $(echo "$FILES" | wc -l) Lean files of Spot's package"
python3 tools/verity/lean/lean_changed.py --build $FILES > "$OUT/build.log" 2>&1; BUILD=$?
echo "BUILD_EXIT $BUILD" >> "$OUT/build.log"
tail -3 "$OUT/build.log" | cut -c1-600
( uv run tools/verity/check/suites.py verity/protocols/proofs/spot/verifier verity-core-sampled-proofs benchmarks/one_stage --fresh > "$OUT/suites.log" 2>&1; echo "SUITES_EXIT $?" >> "$OUT/suites.log" ) &
SUITES=$!
python3 tools/verity/lean/lean_changed.py --records verity/protocols/proofs/spot/verifier/lean > "$OUT/lean.log" 2>&1; LEAN=$?
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
echo "BUILD_EXIT $BUILD SUITES_EXIT $S LEAN_EXIT $LEAN"
[ "$BUILD" = 0 ] && [ "$S" = 0 ]
