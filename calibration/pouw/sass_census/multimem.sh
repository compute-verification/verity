#!/usr/bin/env bash
# The multimem probe (multimem.cu) on one leased RTX PRO 6000: builds for the card's target outside the lease (its card
# record's `arch`, its architecture file nvcc's -gencode: probe.py arch, gencode; another card than an sm_120a one is
# refused before anything is built), records the SASS, then runs under `gpu-lease 1`. Writes multimem.txt and
# multimem.sass to $OUT; the last line is the verdict.
#
#   research run --on MACHINE --cwd source --env GPU_LEASE_WHO=LANE -- bash calibration/pouw/sass_census/multimem.sh
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-$(python3 "$HERE/../../../benchmarks/kernels/machine.py" toolkit 13.0)/bin/nvcc} || { echo "multimem.sh: no CUDA 13.0 (machine.py, above)" >&2; exit 2; }
[ -x "$NVCC" ] || { echo "multimem.sh: no nvcc at $NVCC" >&2; exit 2; }
PROBE="$HERE/../opcode_probe/probe.py"
ARCH=$(python3 "$PROBE" arch) || exit 2
[ "$ARCH" = sm_120a ] || { echo "multimem.sh: the card's target is $ARCH, and multimem.cu is sm_120a's: nothing built" >&2; exit 2; }
GENCODE=$(python3 "$PROBE" gencode) || exit 2
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" -gencode "$GENCODE" -O3 -o "$WORK/multimem" "$HERE/multimem.cu" -lcuda || exit 1
"$(dirname "$NVCC")/cuobjdump" -sass "$WORK/multimem" >"$OUT/multimem.sass"
grep -oE '(LDGMC|REDG|RED|MULTIMEM)[A-Z0-9_.]*' "$OUT/multimem.sass" | sort | uniq -c
gpu-lease 1 --wait --max-min 5 -- "$WORK/multimem" | tee "$OUT/multimem.txt"
exit "${PIPESTATUS[0]}"
