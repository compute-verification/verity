#!/usr/bin/env bash
# Regenerate tests/fixtures/coin_tree_rust.json from the live crate's CoinTree (live/examples/coin_tree_lean.rs, run in a
# flock b684b12 checkout with the SHA-512 patches applied, live/ dropped in as crates/flock-live), then
# tests/fixtures/coin_tree_lean.json from the soundness package's layout of the same tapes (tests/lean/CoinTreeVectors.lean).
# PACKAGE_DIR defaults to this checkout's verifier/lean/soundness; pass another checkout's to keep its Lake build directory
# apart. Each fixture records the sha256 of each file its vectors depend on, which the tests compare with the tree.
#   bash tests/lean/coin_tree.sh FLOCK_CHECKOUT [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
flock=$(cd "$1" && pwd)
pkg=$(cd "${2:-$here/../../verifier/lean/soundness}" && pwd)
live=$(cd "$here/../../live" && pwd)
fixtures=$(cd "$here/../fixtures" && pwd)
rust=$(mktemp)
cp -r "$live/." "$flock/crates/flock-live/"
(cd "$flock" && cargo run -q -p flock-live --features sha512 --example coin_tree_lean -- "$rust" >&2)
python3 - "$rust" "$live" "$fixtures" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, live, fixtures = map(Path, sys.argv[1:])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data["inputs"] = {"backends/flock/live/" + f: sha(live / f) for f in ("src/coin_tree.rs", "examples/coin_tree_lean.rs")}
(fixtures / "coin_tree_rust.json").write_text(json.dumps(data, indent=1) + "\n")
PY
out=$(mktemp)
(cd "$pkg" && lake build FlockSoundness.Discharge.ZkTree.Layout >&2 &&
  lake env lean --run "$here/CoinTreeVectors.lean" "$fixtures/coin_tree_rust.json") > "$out"
python3 - "$out" "$here" "$pkg" "$fixtures" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, here, pkg, fixtures = map(Path, sys.argv[1:])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
data["inputs"] = {
    "backends/flock/tests/lean/CoinTreeVectors.lean": sha(here / "CoinTreeVectors.lean"),
    "backends/flock/tests/fixtures/coin_tree_rust.json": sha(fixtures / "coin_tree_rust.json"),
    **{"backends/flock/verifier/lean/soundness/" + f: sha(pkg / f) for f in (
        "FlockSoundness/Discharge/ZkTree/Layout.lean", "FlockSoundness/Discharge/ZkTree/Defs.lean",
        "FlockSoundness/CROnly/Hm96Defs.lean", "lake-manifest.json", "lean-toolchain")}}
(fixtures / "coin_tree_lean.json").write_text(json.dumps(data, indent=1) + "\n")
PY
rm -f "$rust" "$out"
