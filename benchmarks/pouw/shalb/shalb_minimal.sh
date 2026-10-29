#!/usr/bin/env bash
# Per-function minimality of SHA-512's Σ0, Σ1, σ0 and σ1 halves (shalb_minimal.py) on every core the run is given, and
# the ShaLB numbers (shalb.py) over it. cwd: the source tree's root. Output: $OUT/{minimal.json, minimal.log, shalb.json}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
HERE=benchmarks/pouw/shalb
uv run --no-project --with z3-solver python "$HERE/shalb_minimal.py" --max-k "${MAX_K:-3}" --timeout "${TIMEOUT:-300}" \
  --json "$OUT/minimal.json" | tee "$OUT/minimal.log"
python3 "$HERE/shalb.py" --minimal "$OUT/minimal.json" --json "$OUT/shalb.json" > "$OUT/shalb.txt"
