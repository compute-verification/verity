#!/usr/bin/env bash
# The CPU half of window.sh, on any machine: audit.py on each window under a window.sh OUT, then the harness's verify step on
# each card's bench-before transcript.
#   bash benchmarks/pouw/exhaustion/audit.sh SRC [OUT]
#     SRC is window.sh's OUT, or a tar of its contents (unpacked into OUT, default $RESEARCH_RUN_DIR/out, and audited there).
#   env: PY (default python3; needs numpy), JOBS (audit.py's processes, default nproc).
# Writes <schedule>/audit.json beside each window and verify-before/card<g>.json.  Exit: the first window's audit's, else 1.
# The verify reads the bench transcripts' rows from window.sh's scratch on the window's machine (rows.sh): with AUDIT=0 they are
# kept there until this script, run on that machine, has verified them, and then released; elsewhere the verify is skipped.
set -uo pipefail
exec < /dev/null
src=${1:?SRC}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
PY=${PY:-python3}; JOBS=${JOBS:-$(nproc)}
export PYTHONPATH="$ROOT:$ROOT/verity/core/protocols/pouw${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$src" ]; then
  OUT=$src
else
  OUT=${2:-$RESEARCH_RUN_DIR/out}
  mkdir -p "$OUT" && tar -xf "$src" -C "$OUT" || exit 1
fi
. "$HERE/rows.sh"
rows_claim "$OUT"
kept=$(cat "$OUT/kept-rows" 2>/dev/null) && rows_sweep "$(dirname "$kept")"
rc=
for start in "$OUT"/*/start.json; do
  [ -f "$start" ] || continue
  w=$(dirname "$start")
  "$PY" "$HERE/audit.py" "$w" --bench-before "$OUT/bench-before" --bench-after "$OUT/bench-after" --jobs "$JOBS" < /dev/null
  a=$?; echo "audit $(basename "$w") rc=$a"; rc=${rc:-$a}
done
mkdir -p "$OUT/verify-before"
rows_claim "$OUT"
rows_check "$OUT"; rows=$?
for b in "$OUT"/bench-before/card*/bench.json; do
  [ -f "$b" ] || continue
  g=$(basename "$(dirname "$b")")
  if [ "$rows" != 0 ]; then
    echo "verify $g skipped: its transcript's rows are not on this machine (window.sh with AUDIT=0 keeps them on its own)"
    continue
  fi
  "$PY" "$ROOT/benchmarks/pouw/harness/verify.py" "$b" --out "$OUT/verify-before/$g.json" --no-ledger > "$OUT/verify-before/$g.log" 2>&1 < /dev/null
  echo "verify $g rc=$?"
done
rows_release "$OUT"
exit "${rc:-1}"
