#!/usr/bin/env bash
# Regenerate tests/fixtures/exfiltration_lean.json from the soundness package's `audit_exfiltration`
# (tests/lean/ExfiltrationVectors.lean, whose elaboration also has the kernel check every vector's count).  PACKAGE_DIR
# defaults to this checkout's backends/flock/verifier/lean/soundness; pass another checkout's to keep its Lake build
# directory apart.  The fixture records the sha256 of each file the vectors depend on, which the tests compare with the tree.
#   bash tests/lean/generate.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pkg=$(cd "${1:-$here/../../../../backends/flock/verifier/lean/soundness}" && pwd)
out=$(mktemp)
(cd "$pkg" && lake build FlockSoundness.Audit.Influence >&2 && lake env lean --run "$here/ExfiltrationVectors.lean") > "$out"
python3 - "$out" "$here" "$pkg" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, here, pkg = map(Path, sys.argv[1:])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
soundness = "backends/flock/verifier/lean/soundness/"
data["inputs"] = {"protocols/one_stage/tests/lean/ExfiltrationVectors.lean": sha(here / "ExfiltrationVectors.lean"),
                  **{soundness + f: sha(pkg / f) for f in ("FlockSoundness/Audit/Influence.lean",
                                                           "FlockSoundness/Audit/Law.lean", "lake-manifest.json",
                                                           "lean-toolchain")}}
vectors = data.pop("vectors")
text = json.dumps(data, indent=1)[:-2] + ',\n "vectors": [\n' + ",\n".join(json.dumps(v) for v in vectors) + "\n ]\n}\n"
(here.parent / "fixtures" / "exfiltration_lean.json").write_text(text)
PY
rm -f "$out"
