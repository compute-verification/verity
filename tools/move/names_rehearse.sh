#!/usr/bin/env bash
# names_rehearse.sh BASE TAG [MAPS]: the maps that retire NCI's naming on a tree the layout has moved, start to end.
#   BASE  the commit to run on: a staging's head, or main once the layout is there
#   TAG   names the worktree ../names-TAG beside this checkout, the working directory $NAMES_WORK/TAG (default
#         /tmp/names/TAG) and the branch cursor/training-names-moved-TAG-9af3
#   MAPS  the commit holding the thirteen maps as written on the tree before the layout (default: the commit this
#         branch's regen.py was added on, less one)
#   NAMES_REF  the commit the layout's move ran on, when BASE's history has no commit "Move by tools/move/layout_one_map.toml"
# Regenerates the maps (regen.py, regen_hand_edits.py, both beside this script), commits them, runs the two ordinary
# maps that go before the moves, stages the eleven certified moves (stage.py) and runs the two ordinary maps that go
# after them, one commit a map; each ordinary map is run a second time to show it changes nothing. Writes
# lean-moves.json (the Lean declarations the name maps rename) for the lock records' `--moved`.
# Needs uv at the version layout.py asks for on PATH, and pytest in $PY (default: python3).
# Stops at the first step that fails. Pushes nothing. Not run here: `pins.py --regen` and the Lean records, which
# build Lean and go to a node.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
BASE=$(git -C "$REPO" rev-parse --verify "$1^{commit}")
TAG=$2
MAPS=$(git -C "$REPO" rev-parse --verify "${3:-$(git -C "$REPO" log --format=%H --diff-filter=A -1 HEAD -- tools/move/regen.py)^}^{commit}")
W=${NAMES_WORK:-/tmp/names}/$TAG
WT=$(dirname "$REPO")/names-$TAG
BRANCH=cursor/training-names-moved-$TAG-9af3
PY=${PY:-python3}
mkdir -p "$W/before"
for m in bracket fresh_input fresh_input_prose inference_block_prose inference_only_ids inference_only \
  inference_only_prose slowdown_calculator slowdown_calculator_names slowdown_calculator_terms slowdown slowdown_prose theta; do
  git -C "$REPO" show "$MAPS:tools/move/${m}_map.toml" > "$W/before/${m}_map.toml"
done
echo '[["Security/", "security/"]]' > "$W/security.json"
git -C "$REPO" worktree add -q --detach "$WT" "$BASE"
cd "$WT"

# R. the maps, regenerated from the tree's own layout map; REF is the commit the layout's move ran on
if [ -n "${NAMES_REF:-}" ]; then   # the layout's move isn't a commit of that subject here: name the commit it ran on
  REF=$(git rev-parse --verify "$NAMES_REF^{commit}")
else
  MOVE=$(git log --format=%H --grep='^Move by tools/move/layout_one_map.toml' -1 "$BASE")
  test -n "$MOVE" || { echo "no commit 'Move by tools/move/layout_one_map.toml' below $BASE: set NAMES_REF to the commit the layout's move ran on"; exit 1; }
  REF=$(git rev-parse "$MOVE^")
fi
echo "BASE $BASE; the layout's move ran on REF $REF; the maps before the layout are $MAPS's" | tee "$W/steps.log"
python3 "$HERE/regen.py" --ref "$REF" --moves tools/move/layout_one_map.toml --moves "$W/security.json" \
  --out "$W/regen" "$W"/before/*_map.toml > "$W/regen.log" 2>&1 || { tail -20 "$W/regen.log"; exit 1; }
python3 "$HERE/regen_hand_edits.py" "$W/regen" "$W/moved" tools/move/layout_one_map.toml | tee -a "$W/steps.log"
cp "$W"/moved/*.toml tools/move/
git add tools/move
git commit -qm "tools/move: the inference-only and slowdown maps, regenerated on the moved tree

The maps that retire NCI's naming, their paths rewritten onto the layout's rows by regen.py and regen_hand_edits.py:
the calculator's path map on the moved paths, one path map that flattens the check's directory, and an ordinary map
that names the check's tests by their directory before the moves."
(cd tools/move && "$PY" -m pytest tests/test_maps.py -q -p no:cacheprovider 2>&1 | tail -1) | tee -a "$W/steps.log"

# 0. the ordinary changes before the moves
ordinary() {   # MAP SUBJECT: run it, commit what it changed (nothing to commit is said, not an error), run it again
  python3 tools/move/rename.py --map "tools/move/$1" > "$W/$1.out" 2>&1 || { tail -20 "$W/$1.out"; exit 1; }
  if git diff --quiet HEAD; then   # rename.py stages what it changes
    echo "$1: changes nothing here" | tee -a "$W/steps.log"
  else
    n=$(git diff --name-only HEAD | wc -l)
    git commit -qam "$2

\`python3 tools/move/rename.py --map tools/move/$1\`, an ordinary change."
    python3 tools/move/rename.py --map "tools/move/$1" > "$W/$1.again.out" 2>&1
    git diff --quiet HEAD || { echo "$1 is not idempotent"; exit 1; }
    echo "$1: $n files, $(git rev-parse --short=12 HEAD); run again, it changes nothing" | tee -a "$W/steps.log"
  fi
}
ordinary slowdown_calculator_terms_map.toml "the slowdown calculator: the Glossary's words"
ordinary inference_only_check_tests_map.toml "the inference-only check's README: its tests by their directory"
git rev-parse HEAD > "$W/base.sha"

# 1. the Lean names that move, one file for the four name maps
python3 - <<'EOF'
import json, pathlib, tomllib
M = pathlib.Path("tools/move")
text = (M / "inference_only_map.toml").read_text()
slow, fresh, theta = (tomllib.loads((M / f"{n}.toml").read_text()) for n in ("slowdown_map", "fresh_input_map", "theta_map"))
q = lambda s: json.dumps(s, ensure_ascii=False)
rows = "".join(f"{q(k)} = {q(v)}\n" for d in (slow, fresh, theta) for k, v in d["words"].items())
text = text.replace("[words]\n", "[words]\n" + rows, 1)
text = text.replace("[keep]\npatterns = [", "[keep]\npatterns = [" + "".join(f"'{p}', " for p in fresh["keep"]["patterns"]), 1)
text = text.replace("[keep_paths]\n", "[keep_paths]\n" + "".join(f"{q(g)} = {json.dumps(w)}\n" for g, w in fresh["keep_paths"].items()), 1)
a, b = theta["words_in"]
for files, words in ((fresh["words_in"][0]["files"], fresh["words_in"][0]["words"]), (a["files"] + b["files"], a["words"])):
    text += "\n[[words_in]]\nfiles = " + json.dumps(files) + "\nwords = { " + ", ".join(f"{q(k)} = {q(v)}" for k, v in words.items()) + " }\n"
(M / "zz_names_all_map.toml").write_text(text)
EOF
python3 tools/move/rename.py --map tools/move/zz_names_all_map.toml --moves > "$W/lean-moves.json"
rm tools/move/zz_names_all_map.toml
test -z "$(git status --porcelain)"
echo "lean-moves.json: $(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$W/lean-moves.json") names" | tee -a "$W/steps.log"

# 2. the certified moves
M=tools/move
python3 $M/stage.py $M/bracket_map.toml $M/fresh_input_map.toml $M/theta_map.toml $M/inference_only_map.toml \
  $M/slowdown_map.toml $M/inference_only_prose_map.toml $M/slowdown_prose_map.toml $M/inference_block_prose_map.toml \
  $M/fresh_input_prose_map.toml $M/slowdown_calculator_map.toml $M/inference_only_check_map.toml \
  --base "$(cat "$W/base.sha")" --no-fetch --branch "$BRANCH" --certificate "$W/cert.json" > "$W/stage.log" 2>&1 \
  || { grep -n "stage: " "$W/stage.log" | cut -c1-1200; exit 1; }
grep -c "move-check: certified" "$W/stage.log" | sed 's/$/ moves certified/' | tee -a "$W/steps.log"

# 3. the two ordinary maps after the moves
git checkout -q "$BRANCH"
ordinary inference_only_ids_map.toml "the inference-only policy's ids"
ordinary slowdown_calculator_names_map.toml "the slowdown calculator's names"
echo "head $(git rev-parse HEAD) on $BRANCH" | tee -a "$W/steps.log"
