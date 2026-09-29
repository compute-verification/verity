#!/usr/bin/env bash
# Regenerate tests/fixtures/exfiltration_lean.json from the soundness package's `audit_exfiltration`
# (tests/lean/ExfiltrationVectors.lean).  PACKAGE_DIR defaults to this checkout's
# backends/flock/verifier/lean/soundness; pass another checkout's to keep its Lake build directory apart.
#   bash tests/lean/generate.sh [PACKAGE_DIR]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pkg=${1:-$here/../../../../backends/flock/verifier/lean/soundness}
cd "$pkg"
lake build FlockSoundness.Audit.Influence
lake env lean --run "$here/ExfiltrationVectors.lean" > "$here/../fixtures/exfiltration_lean.json"
