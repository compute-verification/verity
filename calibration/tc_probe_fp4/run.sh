#!/usr/bin/env bash
# verity bench tc_probe_fp4 [ARGS]: probe.py with ARGS (default --instruction nvf4 --sass-check --sweep) into $OUT, built for
# the card the run opened; a card of another architecture than the instruction's kernels, or no GPU, is refused by name.
set -euo pipefail
[ $# -gt 0 ] || set -- --instruction nvf4 --sass-check --sweep
OUT=${OUT:-${RESEARCH_RUN_DIR:-./out}}
mkdir -p "$OUT"
exec python calibration/tc_probe_fp4/probe.py --out "$OUT" "$@"
