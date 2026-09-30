#!/usr/bin/env bash
# Regenerate tests/fixtures/randomness_lean.json from the soundness package's Lean spec of verity.randomness v1 and the
# PoUW window draw (tests/lean/RandomnessVectors.lean). PACKAGE_DIR defaults to this checkout's
# verifier/lean/soundness; pass another checkout's to keep its Lake build directory apart. The fixture records the sha256
# of each file the vectors depend on, which the tests compare with the tree.
#   bash tests/lean/generate.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
pkg=$(cd "${1:-$here/../../verifier/lean/soundness}" && pwd)
out=$(mktemp)
(cd "$pkg" && lake build FlockSoundness.PlanDraw >&2 &&
  lake env lean --run "$here/RandomnessVectors.lean" "$root/packages/verity/tests/randomness/vectors.json") > "$out"
python3 - "$out" "$here" "$pkg" "$root" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, here, pkg, root = map(Path, sys.argv[1:])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
verifier = pkg.parent
data["inputs"] = {
    "backends/flock/tests/lean/RandomnessVectors.lean": sha(here / "RandomnessVectors.lean"),
    "packages/verity/tests/randomness/vectors.json": sha(root / "packages/verity/tests/randomness/vectors.json"),
    **{"backends/flock/verifier/lean/soundness/" + f: sha(pkg / f) for f in (
        "FlockSoundness/Randomness.lean", "FlockSoundness/PlanDraw.lean", "lake-manifest.json", "lean-toolchain")},
    **{"backends/flock/verifier/lean/" + f: sha(verifier / f) for f in ("Flock/Hash.lean", "Flock/Draw.lean")}}
(here.parent / "fixtures" / "randomness_lean.json").write_text(json.dumps(data, indent=1) + "\n")
PY
rm -f "$out"
