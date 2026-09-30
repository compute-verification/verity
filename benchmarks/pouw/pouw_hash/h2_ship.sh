#!/usr/bin/env bash
# The -h2 hash bench's tree: pearl_c_sm120's (build.sh: the cubin and its SASS gate, the check fixture, run.py, toolchain.txt,
# source.txt), h2_bench.py beside run.py, and under py/ the stdlib-only `verity.commitments` it imports (the node's python3
# has neither numpy nor a blake3 package); MANIFEST.sha256 over all of it.
# Usage: bash benchmarks/pouw/pouw_hash/h2_ship.sh OUT   (CUDA names the toolkit, as build.sh's)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd); OUT=${1:?usage: h2_ship.sh OUT}
bash "$ROOT/benchmarks/pouw/pearl_c_sm120/build.sh" "$OUT"
cp "$HERE/h2_bench.py" "$OUT/"
V=$ROOT/packages/verity/src/verity
mkdir -p "$OUT/py/verity/commitments"
cp "$V/__init__.py" "$V/errors.py" "$OUT/py/verity/"
cp "$V"/commitments/*.py "$OUT/py/verity/commitments/"
(cd /tmp && python3 -I -c "import sys; sys.path.insert(0, '$OUT/py'); import verity.commitments.merkle; assert 'numpy' not in sys.modules")
git -C "$ROOT" status --porcelain -- benchmarks/pouw/pouw_hash packages/verity/src/verity/commitments >> "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
