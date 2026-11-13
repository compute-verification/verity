#!/bin/bash
# cross_move.sh: what the move of verity/definitions/ does to the superseded-versions check.
#
# Run in a clone of this branch (the check's head merged with the branch that holds the move's maps), as a remote run:
#   research run --on auto --cwd clone -- bash scratch/CIR-208/cross_move.sh
# It runs the three path maps in their order, each as one commit, and holds the check's four files to the moved tree
# (`checks`): every path string of the list names a file, the test file runs, `--prune` changes nothing but the order, and
# the test file's graph task has no undeclared read. Then it runs the Lake packages' map and holds them again.
# Everything it prints is also under $OUT, a directory a phase.
set -uo pipefail
top=${OUT:?the run directory\'s out/}
here=$(cd "$(dirname "$0")" && pwd)
G="git -c user.name=cir208-scratch -c user.email=cir208-scratch@example.invalid"
LIST=tests/lint/allowlists/superseded_versions.json
TEST=tools/verity/circuit_check/tests/test_superseded_versions.py
MINE="$LIST $TEST tools/verity/circuit_check/pyproject.toml tests/README.md"
# the helpers, copied out of the tree: a map may rewrite a file under scratch/, and the tree must stay clean between maps
tools=$(mktemp -d)
cp "$here/list_paths.py" "$here/same_entries.py" "$here/graph_result.py" "$tools/"
start=$(git rev-parse HEAD)
echo "start $start"

run_maps() {
  for m in "$@"; do
    uv run -q --no-project --python 3.12 python tools/move/layout.py --map tools/move/$m.toml --no-lock > "$top/$m.log" 2>&1
    rc=$?
    echo "$m: layout.py exit $rc; $(tail -1 "$top/$m.log" | cut -c1-200)"
    if [ $rc -ne 0 ]; then tail -40 "$top/$m.log"; echo "STOPPED at $m"; return 1; fi
    $G add -A && $G commit -qm "scratch: tools/move/layout.py --map tools/move/$m.toml --no-lock"
    echo "$m: $(git show --stat --format= HEAD | tail -1)"
  done
}

checks() {
  out=$top/$1
  mkdir -p "$out"
  echo "== the check's four files across the maps"
  git diff --stat "$start" HEAD -- $MINE
  git diff "$start" HEAD -- $MINE > "$out/my-files.diff"
  cp "$LIST" "$out/list.after-move.json"

  echo "== every path string of the list names a file of the moved tree"
  python3 "$tools/list_paths.py" "$LIST" | tee "$out/list-paths.txt"

  echo "== uv sync on the moved tree"
  uv sync -q --all-packages --extra torch-cpu > "$out/uv-sync.log" 2>&1
  echo "uv sync exit $?; $(tail -2 "$out/uv-sync.log" | cut -c1-300)"

  echo "== the test file on the moved tree"
  (cd tools/verity/circuit_check && uv run -q pytest -q -p no:cacheprovider tests/test_superseded_versions.py) > "$out/test.after-move.log" 2>&1
  echo "pytest exit $?"
  tail -40 "$out/test.after-move.log" | cut -c1-600

  echo "== --prune on the moved tree"
  uv run -q python "$TEST" --prune > "$out/prune.log" 2>&1
  echo "prune exit $?; $(tail -3 "$out/prune.log" | cut -c1-300)"
  git diff --stat -- "$LIST"
  python3 "$tools/same_entries.py" "$out/list.after-move.json" "$LIST"
  cp "$LIST" "$out/list.after-prune.json"
  (cd tools/verity/circuit_check && uv run -q pytest -q -p no:cacheprovider tests/test_superseded_versions.py) > "$out/test.after-prune.log" 2>&1
  echo "pytest after --prune exit $?; $(tail -1 "$out/test.after-prune.log")"

  echo "== the test file's graph task on the moved tree (the read guard)"
  work=$(mktemp -d)
  uv run -q verity graph run "test-file:$TEST" --work "$work" > "$out/graph-task.json" 2> "$out/graph-task.err"
  echo "graph exit $?"
  python3 "$tools/graph_result.py" "$out/graph-task.json"
  tail -5 "$out/graph-task.err" | cut -c1-400
  rm -rf "$work"
  # leave the tree as the maps made it: the next map refuses a dirty worktree
  git checkout -q -- . && git status --short | head -5
}

echo "#### the three maps that take verity/definitions/ out of verity/"
run_maps definitions_kept_map definitions_merge_map definitions_served_map && checks three-maps
echo
echo "#### then the Lake packages' map"
run_maps definitions_lake_map && checks four-maps
echo "DONE"
