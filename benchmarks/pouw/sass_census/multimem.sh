#!/usr/bin/env bash
# The multimem probe (multimem.cu) on one leased RTX PRO 6000: builds for sm_120a outside the lease, records the SASS,
# then runs under `gpu-lease 1`. Writes multimem.txt and multimem.sass to $OUT; the last line is the verdict.
#
#   research run --on vy-nebius-1 --cwd source --env GPU_LEASE_WHO=<lane> -- bash benchmarks/pouw/sass_census/multimem.sh
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-$(ls -d /usr/local/cuda-13*/bin/nvcc 2>/dev/null | sort -V | tail -1)}
[ -x "${NVCC:-}" ] || NVCC=$(command -v nvcc)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" -arch=sm_120a -O3 -o "$WORK/multimem" "$HERE/multimem.cu" -lcuda || exit 1
"$(dirname "$NVCC")/cuobjdump" -sass "$WORK/multimem" >"$OUT/multimem.sass"
grep -oE '(LDGMC|REDG|RED|MULTIMEM)[A-Z0-9_.]*' "$OUT/multimem.sass" | sort | uniq -c
gpu-lease 1 --wait --max-min 5 -- "$WORK/multimem" | tee "$OUT/multimem.txt"
exit "${PIPESTATUS[0]}"
