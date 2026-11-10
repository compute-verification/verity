set -eu
# The other way, dry: the remote merge's base with its nine no-row files taken off, crossed by main's layout.py as
# restack.py runs it on a branch (--root CHECKOUT --map MAP --no-lock --report-splits), then the nine put back byte for
# byte at their new paths; and beside it the same base carried by the pull request's layout.py with a rows file, to read
# off what a hand must still change after the put-back. A clone that is thrown away; nothing here is pushed.
export GIT_AUTHOR_NAME=lean GIT_AUTHOR_EMAIL=lean@verity.invalid GIT_COMMITTER_NAME=lean GIT_COMMITTER_EMAIL=lean@verity.invalid
B1=97876e5336e1f0a7c746f8ce0dff6cb0e1f74ade          # the base's head the order names
B2=f63a7b76334dcc8adb0d63de0e35efa3463c5ea3          # the base's head at launch: B1 and one commit
M=a611d18804ac4a22ff518c7b19df717f6ffcaa77           # the map's branch
MAP=tools/move/layout_one_map.toml
ROWS=tools/move/layout_one_map.rows.toml
NAMES='sha512_stream|core/periods|core\.periods|verity_periods|verity/periods|verity\.periods|commitments\.keyed|commitments/keyed|import keyed|from \.keyed|from \. import [a-z_, ]*keyed|commitments import [a-z_, ]*keyed|network/README'
OLDROOTS='verity[./](core|ml)[./]'
S="$OUT/summary.txt"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cp -r .dry2 "$T/in"
say() { echo "$*" | tee -a "$S"; }
say "scratch head $(git rev-parse HEAD); B1 $B1; B2 $B2; map's branch $M"
for f in layout_main.py layout_8711.py rows.toml nine.tsv overlay/tools/move/layout_one_map.toml; do
  say "input $f blob $(git hash-object "$T/in/$f")"
done

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
oldroots() { git grep -nE "$OLDROOTS" -- . ':!tools/move' ':!uv.lock' | sed -E 's/^([^:]*):[0-9]+:/\1: /' | sort > "$1" || true; }
overlay() {   # overlay BASE: BASE with the map's branch's four files, committed
  git checkout -q --detach "$1"
  cp -r "$T/in/overlay/." .
  git add -A
  git commit -q -m "scratch: the map's branch's four files under tools/move"
}
without() {   # the nine taken off, committed
  cut -f1 "$T/in/nine.tsv" | xargs git rm -q --
  git commit -q -m "scratch: without the nine"
}

# 0. the reference: main's layout.py on the map's own tree
git checkout -q --detach "$M"
run "$T/in/layout_main.py" 0-map-tree
oldroots "$OUT/0-old-roots.txt"
back "$M"

# A1. the base at B1 without the nine, main's copy
overlay "$B1"
X1=$(git rev-parse HEAD)
say "X1 = B1 + the map's branch's files: $(git diff --name-only "$B1" "$X1" | tr '\n' ' ')"
without
Z1=$(git rev-parse HEAD)
run "$T/in/layout_main.py" a1-main-without-nine-at-B1
head -c 3000 "$OUT/a1-main-without-nine-at-B1.err" | tee -a "$S"
TA1=$(git write-tree)
back "$Z1"

# A2. the same at B2
overlay "$B2"
X2=$(git rev-parse HEAD)
say "map blob in X2: $(git rev-parse "$X2:$MAP"); layout.py in X2: $(git rev-parse "$X2:tools/move/layout.py")"
git grep -nE "$NAMES" > "$OUT/a2-names-before.txt" || true
say "A2: lines naming the nine on X2: $(wc -l < "$OUT/a2-names-before.txt")"
without
Z2=$(git rev-parse HEAD)
git grep -nE "$NAMES" > "$OUT/a2-names-before-without.txt" || true
run "$T/in/layout_main.py" a2-main-without-nine-at-B2
head -c 3000 "$OUT/a2-main-without-nine-at-B2.err" | tee -a "$S"
TA2=$(git write-tree)
git diff --name-status "$TA1" "$TA2" > "$OUT/a1-vs-a2.name-status.txt" || true
say "A1 against A2: $(wc -l < "$OUT/a1-vs-a2.name-status.txt") paths differ: $(tr '\n\t' ' :' < "$OUT/a1-vs-a2.name-status.txt" | head -c 600)"
TR=
if [ "$RC" = 0 ]; then
  notes a2-main-without-nine-at-B2
  git diff --cached --name-status --no-renames "$Z2" > "$OUT/a2-name-status.txt"
  say "A2: against the tree before the move: added $(grep -c '^A' "$OUT/a2-name-status.txt"), deleted $(grep -c '^D' "$OUT/a2-name-status.txt"), modified $(grep -c '^M' "$OUT/a2-name-status.txt")"
  say "A2: files left under verity/core and verity/ml in the result:"
  git ls-files verity/core verity/ml | tee -a "$S"
  git grep -nE "$NAMES" > "$OUT/a2-names-after.txt" || true
  say "A2: lines naming the nine after: $(wc -l < "$OUT/a2-names-after.txt")"
  oldroots "$OUT/a2-old-roots.txt"
  comm -13 "$OUT/0-old-roots.txt" "$OUT/a2-old-roots.txt" > "$OUT/a2-old-roots-only-here.txt"
  say "A2: lines that still spell verity.core / verity.ml / verity/core / verity/ml outside tools/move and uv.lock: $(wc -l < "$OUT/a2-old-roots.txt"); on the map's own tree after its move: $(wc -l < "$OUT/0-old-roots.txt"); only here: $(wc -l < "$OUT/a2-old-roots-only-here.txt")"
  diff <(grep "^    " "$OUT/0-map-tree.out" | sort) <(grep "^    " "$OUT/a2-main-without-nine-at-B2.out" | sort) > "$OUT/a2-notes-vs-map-tree.diff" || true
  say "A2: note lines only in this run: $(grep -c '^>' "$OUT/a2-notes-vs-map-tree.diff" || true); only on the map's own tree: $(grep -c '^<' "$OUT/a2-notes-vs-map-tree.diff" || true)"
  # the put-back, byte for byte
  while IFS="$(printf '\t')" read -r old new; do
    if [ -e "$new" ]; then say "A2: PUT-BACK CLASH: $new is already there"; continue; fi
    mkdir -p "$(dirname "$new")"
    git show "$X2:$old" > "$new"
    say "A2: put back $old -> $new, blob $(git hash-object "$new") (was $(git rev-parse "$X2:$old"))"
  done < "$T/in/nine.tsv"
  git add -A
  TR=$(git write-tree)
  say "A2: tree with the nine put back: $TR"
  cut -f2 "$T/in/nine.tsv" | xargs grep -nE "$OLDROOTS" > "$OUT/a2-put-back-stale-lines.txt" || true
  say "A2: lines of the nine, put back, that spell an old root: $(wc -l < "$OUT/a2-put-back-stale-lines.txt")"
fi
back "$Z2"

# B. second: the same base carried by the pull request's copy with the rows file
git checkout -q --detach "$X2"
cp "$T/in/rows.toml" $ROWS
git add -A
git commit -q -m "scratch: the rows file"
Y2=$(git rev-parse HEAD)
run "$T/in/layout_8711.py" b-pr-rows-at-B2
head -c 3000 "$OUT/b-pr-rows-at-B2.err" | tee -a "$S"
if [ "$RC" = 0 ]; then
  TB=$(git write-tree)
  notes b-pr-rows-at-B2
  say "B: files left under verity/core and verity/ml in the result:"
  git ls-files verity/core verity/ml | tee -a "$S"
  diff <(grep "^    " "$OUT/0-map-tree.out" | sort) <(grep "^    " "$OUT/b-pr-rows-at-B2.out" | sort) > "$OUT/b-notes-vs-map-tree.diff" || true
  say "B: note lines only in this run: $(grep -c '^>' "$OUT/b-notes-vs-map-tree.diff" || true); only on the map's own tree: $(grep -c '^<' "$OUT/b-notes-vs-map-tree.diff" || true)"
  if [ -n "$TR" ]; then
    git diff --stat=200 "$TR" "$TB" > "$OUT/put-back-vs-carried.stat.txt" || true
    git diff "$TR" "$TB" > "$OUT/put-back-vs-carried.patch" || true
    say "the put-back tree against the carried tree: $(git diff --name-only "$TR" "$TB" | wc -l) paths differ, $(grep -c '^[-+][^-+]' "$OUT/put-back-vs-carried.patch" || true) lines"
    cat "$OUT/put-back-vs-carried.stat.txt" | tee -a "$S"
  fi
fi
back "$Y2"
say "done"
