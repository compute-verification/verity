#!/usr/bin/env bash
# Regenerate verifier/tests/fixtures/exfiltration_lean.json from the proofs package's `audit_exfiltration`
# (verifier/tests/lean/ExfiltrationVectors.lean) and verifier/tests/fixtures/exfiltration_two_lean.json from its two-stage count
# `card_influenceTwo_le_uniform` (verifier/tests/lean/ExfiltrationTwoVectors.lean), each elaboration also having the kernel check
# every vector's count.  PACKAGE_DIR defaults to this checkout's security/Proofs; pass another checkout's to keep its Lake
# build directory apart.  Each fixture records the sha256 of each file its vectors depend on, which the tests compare with
# the tree.
#   bash verifier/tests/lean/generate.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../../../../.." && pwd)
proofs=$root/security/Proofs
pkg=$(cd "${1:-$proofs}" && pwd)
one=$(mktemp)
two=$(mktemp)
# `lean --run` prints a file's errors on stdout, which goes to the vectors' file: on a failure, show it.
(cd "$pkg" && lake build Proofs.Spot.Sampling.Influence Proofs.Spot.Sampling.StratifiedMiss \
  Proofs.Flock.SamplingGeneric.TwoTable >&2 &&
  { lake env lean --run "$here/ExfiltrationVectors.lean" > "$one" || { cat "$one" >&2; exit 1; }; } &&
  { lake env lean --run "$here/ExfiltrationTwoVectors.lean" > "$two" || { cat "$two" >&2; exit 1; }; })
python3 - "$one" "$two" "$here" "$pkg" <<'PY'
import hashlib, json, sys
from pathlib import Path
one, two, here, pkg = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tests = "tests/verity/protocols/proofs/spot/verifier/lean/"
soundness = "security/Proofs/"
pins = ("lake-manifest.json", "lean-toolchain")


def write(out: Path, name: str, inputs: dict, keys: tuple) -> None:
    data = json.loads(out.read_text())
    data["inputs"] = inputs
    lists = {key: data.pop(key) for key in keys}
    text = json.dumps(data, indent=1)[:-2] + "".join(f',\n "{key}": [\n' + ",\n".join(json.dumps(v) for v in vs) + "\n ]"
                                                     for key, vs in lists.items()) + "\n}\n"
    (here.parent / "fixtures" / name).write_text(text)


write(one, "exfiltration_lean.json",
      {tests + "ExfiltrationVectors.lean": sha(here / "ExfiltrationVectors.lean"),
       **{soundness + f: sha(pkg / f) for f in ("Spot/Sampling/Influence.lean", "Spot/Sampling/SamplingDesign.lean",
                                                "Spot/Sampling/Stratified.lean",
                                                "Spot/Sampling/StratifiedMiss.lean", *pins)}},
      ("vectors", "stratified"))
write(two, "exfiltration_two_lean.json",
      {tests + "ExfiltrationTwoVectors.lean": sha(here / "ExfiltrationTwoVectors.lean"),
       "security/Definitions/Claims/SamplingTwo.lean": sha(pkg / "../Definitions/Claims/SamplingTwo.lean"),
       **{soundness + f: sha(pkg / f) for f in ("Flock/SamplingGeneric/TwoCount.lean", "Flock/SamplingGeneric/TwoTable.lean",
                                                *pins)}},
      ("vectors",))
PY
rm -f "$one" "$two"
