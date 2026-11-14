#!/usr/bin/env bash
# Pearl-C's sm_120 ship, built off the card: build.sh's tree (the cubin, refused unless its gate passes), and
#   check/               fixture.py's inputs and expected buffers (needs verity and numpy)
#   run.py, source.txt (the verity commit and the status of the trees the ship is built from), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c_sm120/ship.sh [OUT]   (default /tmp/pearl-c-sm120-ship; CUDA names the toolkit, as for
#        build.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${1:-/tmp/pearl-c-sm120-ship}
bash "$HERE/build.sh" "$OUT"
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c_sm120/fixture.py "$OUT/check")
cp "$HERE/run.py" "$OUT/"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c_sm120 benchmarks/pouw/pearl_c verity/protocols/accounting/pouw/verifier verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
