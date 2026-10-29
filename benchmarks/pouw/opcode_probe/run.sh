#!/usr/bin/env bash
# Build every probe, gate it, then gate and time on one leased GPU; the rows go to $OUT. From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/opcode_probe/run.sh [SOURCE.cu...] [-- probe.py run flags]
#
# The build and the first gate run outside the lease; `probe.py run` gates again on the leased GPU and times nothing
# unless every verdict is the declared one.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
sources=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do sources+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"

python3 "$HERE/probe.py" build --out "$WORK" "${sources[@]}" >"$WORK/built" 2>"$OUT/build.log" \
  || { cat "$OUT/build.log" >&2; exit 1; }
mapfile -t bins <"$WORK/built"
for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
python3 "$HERE/probe.py" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc

gpu-lease 1 --wait --max-min 20 -- python3 "$HERE/probe.py" run "${bins[@]}" --out "$OUT" "$@"
