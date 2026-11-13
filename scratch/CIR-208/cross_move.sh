#!/bin/bash
# cross_move.sh: what the move of verity/definitions/ does to the superseded-versions check.
#
# Run in a clone of this branch (the check's head merged with the branch that holds the move's maps), as a remote run:
#   research run --on auto --cwd clone -- bash scratch/CIR-208/cross_move.sh
# It runs the three path maps in their order, each as one commit, then holds the check's four files to the moved tree:
# every path string of the list names a file, the test file runs, `--prune` changes nothing but the order, and the test
# file's graph task has no undeclared read. Everything it prints is also under $OUT.
set -uo pipefail
out=${OUT:?the run directory\'s out/}
G="git -c user.name=cir208-scratch -c user.email=cir208-scratch@example.invalid"
LIST=tests/lint/allowlists/superseded_versions.json
TEST=tools/verity/circuit_check/tests/test_superseded_versions.py
MINE="$LIST $TEST tools/verity/circuit_check/pyproject.toml tests/README.md"
start=$(git rev-parse HEAD)
echo "start $start"

for m in definitions_kept_map definitions_merge_map definitions_served_map; do
  uv run -q --no-project --python 3.12 python tools/move/layout.py --map tools/move/$m.toml --no-lock > "$out/$m.log" 2>&1
  rc=$?
  echo "$m: layout.py exit $rc; $(tail -1 "$out/$m.log" | cut -c1-200)"
  if [ $rc -ne 0 ]; then tail -40 "$out/$m.log"; echo "STOPPED at $m"; break; fi
  $G add -A && $G commit -qm "scratch: tools/move/layout.py --map tools/move/$m.toml --no-lock"
  echo "$m: $(git show --stat --format= HEAD | tail -1)"
done

echo "== the check's four files across the maps"
git diff --stat "$start" HEAD -- $MINE
git diff "$start" HEAD -- $MINE > "$out/my-files.diff"
cp "$LIST" "$out/list.after-move.json"

echo "== every path string of the list names a file of the moved tree"
python3 - "$LIST" <<'EOF' | tee "$out/list-paths.txt"
import collections, json, sys
from pathlib import Path
doc = json.load(open(sys.argv[1]))
files = collections.Counter(e["file"] for e in doc["entries"])
gone = sorted(f for f in files if not Path(f).is_file())
roots = collections.Counter("/".join(f.split("/")[:4]) for f in files)
print(f"{len(doc['entries'])} entries in {len(files)} files; {len(gone)} name no file")
for f in gone:
    print("  no file:", f)
for r, n in sorted(roots.items()):
    print(f"  {n:3d} files under {r}")
EOF

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
python3 - "$out/list.after-move.json" "$LIST" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
key = lambda d: sorted((e["file"], e["detail"]) for e in d["entries"])
print("after --prune: the same entries" if key(a) == key(b) else "after --prune: the entries differ",
      "| the same order" if a["entries"] == b["entries"] else "| in another order", "| held", b["held"])
EOF
cp "$LIST" "$out/list.after-prune.json"
(cd tools/verity/circuit_check && uv run -q pytest -q -p no:cacheprovider tests/test_superseded_versions.py) > "$out/test.after-prune.log" 2>&1
echo "pytest after --prune exit $?; $(tail -1 "$out/test.after-prune.log")"

echo "== the test file's graph task on the moved tree (the read guard)"
work=$(mktemp -d)
uv run -q verity graph run "test-file:$TEST" --work "$work" > "$out/graph-task.json" 2> "$out/graph-task.err"
echo "graph exit $?"
python3 - "$out/graph-task.json" <<'EOF'
import json, sys
try:
    r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
    print({k: r.get(k) for k in ("status", "detail", "undeclared_reads")})
except Exception as e:
    print("no result:", type(e).__name__, e)
EOF
tail -5 "$out/graph-task.err" | cut -c1-400
rm -rf "$work"
echo "DONE"
