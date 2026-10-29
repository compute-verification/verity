#!/usr/bin/env bash
# Per-function minimality of SHA-512's Σ0, Σ1, σ0 and σ1 halves (shalb_minimal.py) on every core the run is given, and
# the ShaLB numbers (shalb.py) over it. cwd: the source tree's root. Env: ONLY (functions), MIN_K, MAX_K, TIMEOUT and
# CONTROL_TIMEOUT (seconds a query), SOLVER (z3 or bitwuzla), REDO (a queries.jsonl whose unknowns alone are rerun),
# EXTRA (comma-separated EXTRA opcodes: the tuples over OPS and them holding at least one), DONE (a queries.jsonl whose
# decided queries are skipped, to resume a sweep; a file sent with `research run --send` is under inputs/).
# Output: $OUT/{minimal.json, queries.jsonl, minimal.log, shalb.json, shalb.txt}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
HERE=benchmarks/pouw/shalb
python3 "$HERE/shalb.py" --json "$OUT/shalb.json" > "$OUT/shalb.txt"
uv run --no-project --with z3-solver --with bitwuzla python "$HERE/shalb_minimal.py" --min-k "${MIN_K:-1}" \
  --max-k "${MAX_K:-3}" --timeout "${TIMEOUT:-300}" --control-timeout "${CONTROL_TIMEOUT:-2400}" --only "${ONLY:-}" \
  --solver "${SOLVER:-bitwuzla}" ${REDO:+--redo "$REDO" --no-controls} ${EXTRA:+--extra "$EXTRA" --no-controls} \
  ${DONE:+--done "$DONE" --no-controls} \
  --json "$OUT/minimal.json" --jsonl "$OUT/queries.jsonl" | tee "$OUT/minimal.log"
