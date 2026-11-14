set -u
# The carried #9636 at its head: the Lean fast path of the packages it touches. One records run from the lowest of them,
# the SPOT verifier's Lean, which audits it and each package that requires it (the certifier's verifier, security with
# security/Proofs, security/Properties), with --update, so that a record that differs is written and kept; then the try
# of each Lean file named in verity/protocols/proofs/spot/verifier/lean/Spot/Registration.lean verity/protocols/proofs/spot/verifier/lean/Spot/Record.lean verity/protocols/proofs/spot/verifier/lean/Spot/OneStage.lean verity/protocols/proofs/spot/verifier/lean/Spot/Run.lean verity/protocols/proofs/spot/verifier/lean/Spot/Pieces.lean security/Properties/Spot.lean security/Proofs/Properties/Spot.lean security/Proofs/Properties/Spot/Pinned.lean security/Proofs/Flock/SpotFlock/Code.lean with the sibling check, on the slot tree the records run built.
git grep -q CHECK_SLOT_BOOK_GB HEAD -- tools/verity/check/settings.py || unset CHECK_SLOT_BOOK_GB
BASE=@BASE@
echo "head $(git rev-parse HEAD)"
git merge-base --is-ancestor "$BASE" HEAD && echo "holds main $BASE" || { echo "DOES NOT HOLD main $BASE"; exit 8; }
git status --short | head -5
python3 tools/verity/lean/lean_changed.py --records verity/protocols/proofs/spot/verifier/lean --update > "$OUT/lean.log" 2>&1; LEAN=$?
echo "LEAN_EXIT $LEAN" >> "$OUT/lean.log"
tail -40 "$OUT/lean.log" | cut -c1-500
RESEARCH_RUN_DIR="${RESEARCH_RUN_DIR:-}"
if [ -n "$RESEARCH_RUN_DIR" ] && [ -d "$RESEARCH_RUN_DIR/lean-audit" ]; then
  mkdir -p "$OUT/lean-audit"
  for f in result.json record allowed-record; do
    [ -e "$RESEARCH_RUN_DIR/lean-audit/$f" ] && cp -a "$RESEARCH_RUN_DIR/lean-audit/$f" "$OUT/lean-audit/"
  done
  ( cd "$RESEARCH_RUN_DIR/lean-audit" && find . -maxdepth 6 \( -name lean-audit.json -o -name report.json -o -name result.json \) | sort | head -200 ) > "$OUT/lean-audit/files.txt" 2>&1
  # each audited package's lock as the run wrote it, whether or not it differs from the committed one
  ( cd "$RESEARCH_RUN_DIR/lean-audit" && for d in audit-*; do [ -d "$d" ] && mkdir -p "$OUT/lean-audit/$d" && find "$d" -maxdepth 5 \( -name lean-audit.json -o -name report.json \) -exec cp --parents {} "$OUT/lean-audit/" \; ; done ) 2>/dev/null
fi
TRY=0; N=0
for F in verity/protocols/proofs/spot/verifier/lean/Spot/Registration.lean verity/protocols/proofs/spot/verifier/lean/Spot/Record.lean verity/protocols/proofs/spot/verifier/lean/Spot/OneStage.lean verity/protocols/proofs/spot/verifier/lean/Spot/Run.lean verity/protocols/proofs/spot/verifier/lean/Spot/Pieces.lean security/Properties/Spot.lean security/Proofs/Properties/Spot.lean security/Proofs/Properties/Spot/Pinned.lean security/Proofs/Flock/SpotFlock/Code.lean; do
  N=$((N + 1)); L="$OUT/try-$N.log"
  echo "TRY $F" > "$L"
  python3 tools/verity/lean/lean_changed.py --try "$F" --siblings >> "$L" 2>&1; T=$?
  echo "TRY_EXIT $T" >> "$L"
  echo "try $N $F: exit $T | $(grep -v '^TRY_EXIT' "$L" | tail -1 | cut -c1-300)"
  [ "$T" = 0 ] || TRY=1
done
git status --short > "$OUT/tree.txt" 2>&1
git diff > "$OUT/tree.patch" 2>&1
echo "TREE_CHANGED_FILES $(wc -l < "$OUT/tree.txt")"
echo "LEAN_EXIT $LEAN TRIES $N TRY_FAILED $TRY"
[ "$LEAN" = 0 ] && [ "$TRY" = 0 ]
