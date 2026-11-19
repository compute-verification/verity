#!/usr/bin/env bash
# verity bench tc_probe ARGS: tc_probe.py with ARGS into $OUT, built for the card the run opened; an instruction of
# another architecture, or a card no instruction is for, is refused by name.
set -euo pipefail
[ $# -gt 0 ] || { echo "tc_probe: name the --instruction (tc_probe.py's table), one of the run's card's architecture" >&2; exit 2; }
OUT=${OUT:-${RESEARCH_RUN_DIR:-./out}}
mkdir -p "$OUT"
exec python calibration/tc_probe/tc_probe.py --out "$OUT" "$@"
