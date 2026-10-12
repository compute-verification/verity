#!/usr/bin/env bash
# Regenerate tests/fixtures/randomness_lean.json from the security proofs package's Lean spec of verity.ml.randomness v2 and the
# PoUW window draw (tests/lean/RandomnessVectors.lean). PACKAGE_DIR defaults to this checkout's
# Security/Proofs; pass another checkout's to keep its Lake build directory apart. The fixture records the sha256
# of each file the vectors depend on, which the tests compare with the tree.
#   bash tests/lean/generate.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../../.." && pwd)
pkg=$(cd "${1:-$root/Security/Proofs}" && pwd)
out=$(mktemp)
(cd "$pkg" && lake build Proofs.Flock.Soundness.PlanDraw >&2 &&
  lake env lean --run "$here/RandomnessVectors.lean" "$root/verity/ml/randomness/tests/vectors.json") > "$out"
python3 - "$out" "$here" "$pkg" "$root" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, here, pkg, root = map(Path, sys.argv[1:])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
verifier = root / "verity/core/service/flock"
data["inputs"] = {
    "verity/ml/flock/tests/lean/RandomnessVectors.lean": sha(here / "RandomnessVectors.lean"),
    "verity/ml/randomness/tests/vectors.json": sha(root / "verity/ml/randomness/tests/vectors.json"),
    "verity/core/primitives/randomness/Randomness.lean":
        sha(root / "verity/core/primitives/randomness/Randomness.lean"),
    **{"Security/Proofs/" + f: sha(pkg / f) for f in (
        "Flock/Soundness/PlanDraw.lean", "lake-manifest.json", "lean-toolchain")},
    **{"verity/core/service/flock/" + f: sha(verifier / f) for f in ("Flock/Hash.lean", "Flock/Draw.lean")}}
(here.parent / "fixtures" / "randomness_lean.json").write_text(json.dumps(data, indent=1) + "\n")
PY
rm -f "$out"
