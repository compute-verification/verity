#!/usr/bin/env bash
# The wipe's launcher run on a processor, one honest run: a rehearsal that certifies nothing (the memory attester is this
# machine's under a key made for the run, the fill is in host memory, and `gpu-lease` is a stand-in that grants at once), for
# finding what breaks. From the tree's root, with `uv sync --all-packages --extra torch-cpu` done and elan on the PATH:
#
#   bash benchmarks/pous/smoke/wipe.sh
#
# The honest run is `served_wipe.sh --run-keys`'s own, which builds `wipe-protocol` in a scratch copy, runs the wipe and ends
# with the line `WIPE {...}`: its verdict and the SHA-512 of the executable that decided. Exit 0 where the verdict is accept.
set -o pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
export PATH="$HOME/.elan/bin:$PATH" DEVELOPMENT=1 PY=${PY:-$ROOT/.venv/bin/python}
FB=$(mktemp -d); SERVE=$(mktemp -d); DIR=$SERVE/out
trap 'rm -rf "$FB" "$SERVE"' EXIT
printf '#!/usr/bin/env bash\nwhile [ "$1" != -- ]; do shift; done\nshift\nGPU_LEASE_UUID=GPU-standin exec "$@"\n' > "$FB/gpu-lease"
chmod +x "$FB/gpu-lease"
echo "### honest: served_wipe.sh, host store, python streamer, run keys"
PATH="$FB:$PATH" bash benchmarks/pous/streamed_wipe/served_wipe.sh "$SERVE" 0 "$DIR" '{"run_keys": true, "store": "host", "streamer": "python"}' \
  | tee "$SERVE/said.txt" | tail -1 | cut -c1-600
grep -q '"verdict": "accept"' "$SERVE/said.txt"
