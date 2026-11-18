#!/usr/bin/env bash
# Pearl-C's sm_120 ship, built off the card: build.sh's tree (the cubin, refused unless its gate passes), and
#   check/               fixture.py's inputs and expected buffers (needs verity and numpy)
#   run.py, source.txt (the verity commit and the status of the trees the ship is built from), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c_sm120/ship.sh [OUT]   (default /tmp/pearl-c-sm120-ship; the toolkit CUDA_HOME, else the
#        machine's 13.0, benchmarks/kernels/machine.py's)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${1:-/tmp/pearl-c-sm120-ship}
CUDA_HOME=$(python3 "$ROOT/benchmarks/kernels/machine.py" toolkit 13.0); export CUDA_HOME
GENCODE=$(python3 "$ROOT/benchmarks/kernels/machine.py" gencode sm_120a); export GENCODE
bash "$HERE/../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120/build.sh" "$OUT"
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c_sm120/fixture.py "$OUT/check")
cp "$HERE/../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120/run.py" "$OUT/"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c_sm120 verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c verity/protocols/resource_accounting/compute/pouw/verifier verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
