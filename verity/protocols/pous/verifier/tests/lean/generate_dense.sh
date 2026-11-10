#!/usr/bin/env bash
# Regenerate tests/fixtures/lean_dense_vectors.json from the Lean model.  Needs a Lake package with the POUS trusted
# layer (`lean/pous`) plus, as an extra library, `PousSponge.SpongeModel` (lean/submissions/sponge-dense/SpongeModel.lean),
# built with `lake build PousSponge.SpongeModel`.  Evaluation takes a few seconds; watch memory on larger cases.
#   bash tests/lean/generate_dense.sh LAKE_PACKAGE_DIR
set -euo pipefail
pkg=${1:?usage: generate_dense.sh LAKE_PACKAGE_DIR}
here=$(cd "$(dirname "$0")" && pwd)
cd "$pkg"
cp "$here/PousDenseCheck.lean" PousDenseCheck.lean
lake env lean --run PousDenseCheck.lean | python3 -c '
import json, sys
data = json.loads(sys.stdin.read())
data = {"source": "tests/lean/PousDenseCheck.lean via tests/lean/generate_dense.sh", **data}
json.dump(data, open(sys.argv[1], "w"), indent=1)
' "$here/../fixtures/lean_dense_vectors.json"
