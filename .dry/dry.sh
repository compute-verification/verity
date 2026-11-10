set -eu
# A dry crossing of tonight's map by the remote merge's base, as restack.py runs layout.py on a branch
# (--root CHECKOUT --map MAP --no-lock --report-splits), in a clone that is thrown away. HEAD is a scratch commit whose
# tree is the base's plus .dry/ (the inputs); nothing here is pushed.
export GIT_AUTHOR_NAME=lean GIT_AUTHOR_EMAIL=lean@verity.invalid GIT_COMMITTER_NAME=lean GIT_COMMITTER_EMAIL=lean@verity.invalid
B=2937654e2146944731401454d4f87f3bd1a1dafe          # the base
M=a611d18804ac4a22ff518c7b19df717f6ffcaa77          # the map's branch
MAP=tools/move/layout_one_map.toml
ROWS=tools/move/layout_one_map.rows.toml
NINE="verity/core/periods/__init__.py verity/core/periods/pyproject.toml verity/core/periods/tests/test_periods.py verity/core/periods/tests/test_periods_difftest.py verity/core/protocols/network/README.md verity/ml/commitments/gates/sha512_stream.circuit-check.json verity/ml/commitments/gates/sha512_stream.circuit-check.py verity/ml/commitments/gates/sha512_stream.py verity/ml/commitments/keyed.py"
NAMES='sha512_stream|core/periods|core\.periods|verity_periods|commitments\.keyed|commitments/keyed|import keyed|from \.keyed|from \. import [a-z_, ]*keyed|commitments import [a-z_, ]*keyed|protocols/network/README'
S="$OUT/summary.txt"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cp -r .dry "$T/in"
say() { echo "$*" | tee -a "$S"; }
say "scratch head $(git rev-parse HEAD); base $B; map's branch $M"
for f in layout_main.py layout_8711.py rows.toml overlay/tools/move/layout_one_map.toml; do
  say "input $f blob $(git hash-object "$T/in/$f")"
done
git checkout -q --detach "$B"
say "base checked out: $(git rev-parse HEAD), dirty $(git status --porcelain | wc -l)"
cp -r "$T/in/overlay/." .
git add -A
git commit -q -m "scratch: the map's branch's four files under tools/move"
X=$(git rev-parse HEAD)
say "X = base + the map's branch's files: $(git diff --name-only "$B" "$X" | tr '\n' ' ')"
say "map blob in X: $(git rev-parse "$X:$MAP")"

run() {   # run SCRIPT LABEL
  set +e
  s=$(date +%s)
  python3 "$1" --root "$PWD" --map $MAP --no-lock --report-splits > "$OUT/$2.out" 2> "$OUT/$2.err"
  rc=$?
  set -e
  git add -A
  say "$2: exit $rc, $(( $(date +%s) - s )) s, tree $(git write-tree), paths changed $(git status --porcelain | wc -l), stderr lines $(wc -l < "$OUT/$2.err")"
  RC=$rc
}
back() { git reset -q --hard "$1"; git clean -fdq; }
notes() { grep -v "^    " "$OUT/$1.out" | tee -a "$S" || true; }

# 0. the reference: main's layout.py on the map's own tree
if git cat-file -e "$M^{commit}" 2>/dev/null; then
  git checkout -q --detach "$M"
  run "$T/in/layout_main.py" 0-map-tree
  notes 0-map-tree
  back "$M"
  git checkout -q --detach "$X"
else
  say "0: the clone lacks $M; no reference run"
fi

# 1. main's copy, no rows file
run "$T/in/layout_main.py" 1-main
head -c 3000 "$OUT/1-main.err" | tee -a "$S"
back "$X"

# 2. the pull request's copy, no rows file
run "$T/in/layout_8711.py" 2-pr
head -c 3000 "$OUT/2-pr.err" | tee -a "$S"
back "$X"

# 3. the pull request's copy with the rows file; a stop that names one file is recorded, that file is taken out of the
#    scratch tree, and the run is made again, so that every stop shows
cp "$T/in/rows.toml" $ROWS
git add -A
git commit -q -m "scratch: the rows file"
Y=$(git rev-parse HEAD)
i=0
while :; do
  i=$((i + 1))
  run "$T/in/layout_8711.py" "3-pr-rows-$i"
  [ "$RC" = 0 ] && break
  head -c 3000 "$OUT/3-pr-rows-$i.err" | tee -a "$S"
  first=$(head -1 "$OUT/3-pr-rows-$i.err")
  back "$Y"
  [ "$i" -ge 6 ] && { say "3: six stops, no more tries"; break; }
  case "$first" in
    *": the rewrite would change a guarantee or reads record"|*": the rewrite would change a pinned digest")
      new=${first%%: the rewrite would change*}
      old=$(python3 - "$T/in/layout_8711.py" "$new" <<'PY'
import importlib.util, subprocess, sys, tomllib
spec = importlib.util.spec_from_file_location("layout", sys.argv[1]); L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
data = tomllib.loads(open("tools/move/layout_one_map.toml").read())
rows = tomllib.loads(open("tools/move/layout_one_map.rows.toml").read())["paths"]
lay = L.Layout({**data, "paths": {**data["paths"], **rows}})
tracked = [p for p in subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True).stdout.split("\0") if p]
print(next((p for p in tracked if lay.file(p) == sys.argv[2]), sys.argv[2]))
PY
)
      say "3: stop $i names $new (before the move: $old); taken out of the scratch tree to see the next stop"
      git rm -q "$old"
      git commit -q -m "scratch: without $old"
      Y=$(git rev-parse HEAD)
      ;;
    *) say "3: stop $i is not one this script goes past"; break ;;
  esac
done
if [ "$RC" = 0 ]; then
  notes "3-pr-rows-$i"
  say "3: files left under verity/core and verity/ml in the result:"
  git ls-files verity/core verity/ml | tee -a "$S"
  say "3: the rows file in the result: $( [ -e $ROWS ] && echo STILL THERE || echo gone )"
  say "3: where the nine are now:"
  git ls-files | grep -E "sha512_stream|/keyed\.py$|verity/periods/|certifier/verifier/network/README.md" | tee -a "$S"
  git grep -nE "$NAMES" > "$OUT/3-names-after.txt" || true
  say "3: lines naming the nine after: $(wc -l < "$OUT/3-names-after.txt")"
  git diff --cached --name-status --no-renames "$Y" > "$OUT/3-name-status.txt"
  say "3: against the tree before the move: added $(grep -c '^A' "$OUT/3-name-status.txt"), deleted $(grep -c '^D' "$OUT/3-name-status.txt"), modified $(grep -c '^M' "$OUT/3-name-status.txt")"
  if [ -s "$OUT/0-map-tree.out" ]; then
    diff <(grep "^    " "$OUT/0-map-tree.out" | sort) <(grep "^    " "$OUT/3-pr-rows-$i.out" | sort) > "$OUT/3-notes-vs-map-tree.diff" || true
    say "3: note lines only in this run: $(grep -c '^>' "$OUT/3-notes-vs-map-tree.diff" || true); only on the map's own tree: $(grep -c '^<' "$OUT/3-notes-vs-map-tree.diff" || true)"
  fi
fi
back "$X"

# 4. the other way, beyond what was asked, as a check of the reading: the nine deleted, main's copy
git grep -nE "$NAMES" > "$OUT/4-names-before.txt" || true
say "4: lines naming the nine on X: $(wc -l < "$OUT/4-names-before.txt")"
git rm -q $NINE
git commit -q -m "scratch: without the nine"
Z=$(git rev-parse HEAD)
git grep -nE "$NAMES" > "$OUT/4-names-before-without.txt" || true
run "$T/in/layout_main.py" 4-main-without-nine
head -c 3000 "$OUT/4-main-without-nine.err" | tee -a "$S"
if [ "$RC" = 0 ]; then
  notes 4-main-without-nine
  git grep -nE "$NAMES" > "$OUT/4-names-after.txt" || true
  say "4: lines naming the nine after: $(wc -l < "$OUT/4-names-after.txt")"
  say "4: files left under verity/core and verity/ml in the result:"
  git ls-files verity/core verity/ml | tee -a "$S"
  if [ -s "$OUT/0-map-tree.out" ]; then
    diff <(grep "^    " "$OUT/0-map-tree.out" | sort) <(grep "^    " "$OUT/4-main-without-nine.out" | sort) > "$OUT/4-notes-vs-map-tree.diff" || true
    say "4: note lines only in this run: $(grep -c '^>' "$OUT/4-notes-vs-map-tree.diff" || true); only on the map's own tree: $(grep -c '^<' "$OUT/4-notes-vs-map-tree.diff" || true)"
  fi
fi
back "$Z"
say "done"
