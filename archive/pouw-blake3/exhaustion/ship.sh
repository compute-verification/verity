#!/usr/bin/env bash
# The window's cubin ship from this tree, on a machine with CUDA >= 12.8 but no workspace venv (node 2): pearl_c_sm120/build.sh
# with the harness's python (jobs/env.sh) standing in for its `uv run python`, then the tree as OUT/ship.tar (one directory,
# ship/) and its SHA-256 in OUT/ship.sha256.
#   bash benchmarks/pouw/exhaustion/ship.sh HARNESS_BUILD [OUT]   (OUT default $RESEARCH_RUN_DIR/out; env CUDA, default
#                                                                  /usr/local/cuda-13.0)
set -euo pipefail
exec < /dev/null
BUILD=${1:?HARNESS_BUILD}; OUT=${2:-$RESEARCH_RUN_DIR/out}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
export CUDA=${CUDA:-/usr/local/cuda-13.0}
. "$ROOT/benchmarks/pouw/harness/jobs/env.sh" "$BUILD"
SCR=$(mktemp -d)
trap 'rm -rf "$SCR"' EXIT
mkdir -p "$SCR/bin" "$OUT"
printf '#!/usr/bin/env bash\n[ "$1" = run ] && shift; [ "$1" = python ] && shift; exec "%s" "$@"\n' "$PY" > "$SCR/bin/uv"
chmod +x "$SCR/bin/uv"
export PATH="$SCR/bin:$CUDA/bin:$PATH" PYTHONPATH="$ROOT/benchmarks/pouw:$PYTHONPATH"
bash "$ROOT/benchmarks/pouw/pearl_c_sm120/build.sh" "$SCR/ship"
tar -cf "$OUT/ship.tar" -C "$SCR" ship
sha256sum "$SCR/ship/pearl_c_sm120.cubin" "$OUT/ship.tar" | tee "$OUT/ship.sha256"
cp "$SCR/ship/sass.json" "$SCR/ship/toolchain.txt" "$SCR/ship/source.txt" "$OUT/"
