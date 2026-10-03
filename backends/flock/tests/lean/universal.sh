#!/usr/bin/env bash
# Regenerate tests/fixtures/universal_lean.json from the soundness package's interpreter of the universal unit
# (FlockSoundness/Discharge/PrivateCircuit/Interp.lean, through tests/lean/UniversalVectors.lean). PACKAGE_DIR defaults to
# this checkout's verifier/lean/soundness; pass another checkout's to keep its Lake build directory apart. The fixture
# records the sha256 of each file the vectors depend on, which the tests compare with the tree.
#   bash tests/lean/universal.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pkg=$(cd "${1:-$here/../../verifier/lean/soundness}" && pwd)
out=$(mktemp)
(cd "$pkg" && lake build FlockSoundness.Discharge.PrivateCircuit.Interp >&2 &&
  lake env lean --run "$here/UniversalVectors.lean") > "$out"
python3 - "$out" "$here" "$pkg" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, here, pkg = map(Path, sys.argv[1:])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data["inputs"] = {
    "backends/flock/tests/lean/UniversalVectors.lean": sha(here / "UniversalVectors.lean"),
    **{"backends/flock/verifier/lean/soundness/" + f: sha(pkg / f) for f in (
        "FlockSoundness/Discharge/PrivateCircuit/Interp.lean", "lake-manifest.json", "lean-toolchain")}}
(here.parent / "fixtures" / "universal_lean.json").write_text(json.dumps(data, indent=1) + "\n")
PY
rm -f "$out"
