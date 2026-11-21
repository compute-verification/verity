#!/usr/bin/env bash
# A Pearl-C build's ship, built off the card: build.sh's tree (each binary refused unless its gate passes), and
#   check/               fixture.py's inputs and expected buffers (needs verity and numpy)
#   run.py and driver.py, source.txt (the verity commit and the status of the trees the ship is built from), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c_ship.sh BUILD [OUT]   (BUILD pearl_c_sm90 or pearl_c_sm120, each also its own
#        pearl_c_*/ship.sh; default /tmp/pearl-c-sm90-ship or /tmp/pearl-c-sm120-ship; the toolkit CUDA_HOME, else the
#        machine's 13.0, benchmarks/kernels/machine.py's)
set -euo pipefail
BUILD=${1:?a build: pearl_c_sm90 or pearl_c_sm120}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
KERNEL=$ROOT/verity/protocols/accounting/compute/pouw/prover/kernels/$BUILD
[ -f "$KERNEL/build.sh" ] && [ -f "$HERE/$BUILD/fixture.py" ] \
  || { echo "pearl_c_ship.sh: $BUILD has no build.sh beside its kernel or no fixture.py here; a build is pearl_c_sm90 or pearl_c_sm120" >&2; exit 2; }
OUT=${2:-/tmp/${BUILD//_/-}-ship}
CUDA_HOME=$(python3 "$ROOT/benchmarks/kernels/machine.py" toolkit 13.0); export CUDA_HOME
GENCODE=$(python3 "$ROOT/benchmarks/kernels/machine.py" gencode "$(bash "$KERNEL/build.sh" --arch)"); export GENCODE
bash "$KERNEL/build.sh" "$OUT"
(cd "$ROOT" && uv run python "benchmarks/pouw/$BUILD/fixture.py" "$OUT/check")
cp "$KERNEL/run.py" "$KERNEL/../pearl_c/driver.py" "$OUT/"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- "benchmarks/pouw/$BUILD" verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
