#!/usr/bin/env bash
# Regenerate tests/fixtures/inner_claims.json: what the inner verifier's algebra reads from the statement, per rep of the
# flat GEMM selftest's honest session (the fetched fixture hidden-outputs/flat/gemm-coordinate), from the Lean verifier of
# record (tests/lean/InnerClaims.lean, interpreted: about half an hour). PACKAGE_DIR defaults to this checkout's
# ../../core/service/flock; pass another checkout's to keep its Lake build directory apart. The fixture records the sha256 of each file
# the dump depends on, which the tests compare with the tree.
#   bash tests/lean/generate_inner_claims.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pkg=$(cd "${1:-$here/../../../../core/service/flock}" && pwd)
set_dir="$here/../fixtures/hidden-outputs/flat/gemm-coordinate"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 "$here/../../verifier/selftest_records.py" "$set_dir/records-cpu" "$work/sessions" >&2
(cd "$pkg" && lake build Flock >&2 &&
  lake env lean --run "$here/InnerClaims.lean" verity/flock-circuit@f5df3bbc+seed-injection "$set_dir/stage/circuit.txt" \
    "$set_dir/stage/pub-4.bin" "$work/sessions/honest") > "$work/dump.json"
python3 - "$work/dump.json" "$here" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, here = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(out.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
s = "verity/ml/flock/tests/fixtures/hidden-outputs/flat/gemm-coordinate/"
data["inputs"] = {
    "verity/ml/flock/tests/lean/InnerClaims.lean": sha(here / "InnerClaims.lean"),
    **{s + f: sha(here.parent / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate" / f) for f in (
        "stage/circuit.txt", "stage/pub-4.bin", "records-cpu/honest.json", "records-cpu/honest.rep0.proof",
        "records-cpu/honest.rep1.proof")}}
(here.parent / "fixtures" / "inner_claims.json").write_text(json.dumps(data, indent=1) + "\n")
PY
