#!/usr/bin/env bash
# Regenerate tests/fixtures/lean_vectors.json from the Lean model.  Needs a Lake package with the POUS trusted layer
# (`lean/pous` of the POUS project) plus, as extra libraries, `Pebbling.Encoding` (p3-forward/Encoding.lean),
# `Pebbling.ColumnAwareDraft`, `PebblingCol.{BandComb,BandWeights,TwoWay,BandTwoWay}` and `PebblingMeets.P3Meets`,
# built with `lake build PebblingMeets.P3Meets`.  Lean evaluates the labels lazily, so cost grows fast with n and wc:
# each case runs alone, and cases that exceed the timeout are dropped.
#   bash tests/lean/generate.sh LAKE_PACKAGE_DIR
set -euo pipefail
pkg=${1:?usage: generate.sh LAKE_PACKAGE_DIR}
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
cd "$pkg"
for case in "(1, 3, 1, 7)" "(2, 2, 1, 5)" "(2, 5, 12, 5)" "(3, 2, 1, 2)" "(3, 3, 2, 8)" "game"; do
  python3 - "$here/PousRefCheck.lean" "$case" > Case.lean <<'PY'
import sys
src = open(sys.argv[1]).read()
a, b = src.index("def encodeCases"), src.index("end PousRefCheck")
enc = "[]" if sys.argv[2] == "game" else f"[{sys.argv[2]}]"
games = src[src.index("def gameCases"):b] if sys.argv[2] == "game" else "def gameCases : List (ℕ × ℕ × ℕ × ℕ) := []\n\n"
print(src[:a] + f"def encodeCases : List (ℕ × ℕ × ℕ × ℕ) := {enc}\n\n" + games + src[b:])
PY
  timeout 900 lake env lean --run Case.lean > "$out/$(echo "$case" | tr -dc '0-9a-z').json" || echo "dropped $case"
done
python3 - "$out" "$here/../fixtures/lean_vectors.json" <<'PY'
import glob, json, sys
enc, game = [], []
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    txt = open(f).read().strip()
    if txt:
        data = json.loads(txt)
        enc += data["encode"]
        game += data["game"]
enc.sort(key=lambda c: (c["n"], c["wc"], c["k"]))
json.dump({"source": "tests/lean/PousRefCheck.lean via tests/lean/generate.sh", "encode": enc, "game": game},
          open(sys.argv[2], "w"), indent=1)
PY
