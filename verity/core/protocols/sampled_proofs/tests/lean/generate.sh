#!/usr/bin/env bash
# Regenerate tests/fixtures/exfiltration_lean.json from the proofs package's `audit_exfiltration`
# (tests/lean/ExfiltrationVectors.lean, whose elaboration also has the kernel check every vector's count).  PACKAGE_DIR
# defaults to this checkout's Security/Proofs; pass another checkout's to keep its Lake build
# directory apart.  The fixture records the sha256 of each file the vectors depend on, which the tests compare with the tree.
#   bash tests/lean/generate.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pkg=$(cd "${1:-$here/../../../../../../Security/Proofs}" && pwd)
out=$(mktemp)
(cd "$pkg" && lake build Proofs.Flock.Soundness.Audit.Influence Proofs.Flock.Soundness.Audit.StratifiedMiss >&2 && lake env lean --run "$here/ExfiltrationVectors.lean") > "$out"
python3 - "$out" "$here" "$pkg" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, here, pkg = map(Path, sys.argv[1:])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
soundness = "Security/Proofs/"
data["inputs"] = {"verity/core/protocols/sampled_proofs/tests/lean/ExfiltrationVectors.lean": sha(here / "ExfiltrationVectors.lean"),
                  **{soundness + f: sha(pkg / f) for f in ("Flock/Soundness/Audit/Influence.lean",
                                                           "Flock/Soundness/Audit/Law.lean",
                                                           "Flock/Soundness/Audit/Stratified.lean",
                                                           "Flock/Soundness/Audit/StratifiedMiss.lean",
                                                           "lake-manifest.json", "lean-toolchain")}}
lists = {key: data.pop(key) for key in ("vectors", "stratified")}
text = json.dumps(data, indent=1)[:-2] + "".join(f',\n "{key}": [\n' + ",\n".join(json.dumps(v) for v in vs) + "\n ]"
                                                 for key, vs in lists.items()) + "\n}\n"
(here.parent / "fixtures" / "exfiltration_lean.json").write_text(text)
PY
rm -f "$out"
