#!/usr/bin/env bash
# The CPU half of window.sh, on any machine: audit.py on each window under a window.sh OUT, then the harness's verify step on
# each card's bench-before transcript.
#   bash benchmarks/pouw/exhaustion/audit.sh SRC [OUT]
#     SRC is window.sh's OUT, or a tar of its contents (unpacked into OUT, default $RESEARCH_RUN_DIR/out, and audited there).
#   env: PY (default python3; needs numpy), JOBS (audit.py's processes, default nproc).
# Writes <schedule>/audit.json beside each window and verify-before/card<g>.json.  Exit: the first window's audit's, else 1.
set -uo pipefail
exec < /dev/null
src=${1:?SRC}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
PY=${PY:-python3}; JOBS=${JOBS:-$(nproc)}
export PYTHONPATH="$ROOT/packages/verity/src:$ROOT/protocols/pouw${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$src" ]; then
  OUT=$src
else
  OUT=${2:-$RESEARCH_RUN_DIR/out}
  mkdir -p "$OUT" && tar -xf "$src" -C "$OUT" || exit 1
fi
rc=
for start in "$OUT"/*/start.json; do
  [ -f "$start" ] || continue
  w=$(dirname "$start")
  "$PY" "$HERE/audit.py" "$w" --bench-before "$OUT/bench-before" --bench-after "$OUT/bench-after" --jobs "$JOBS" < /dev/null
  a=$?; echo "audit $(basename "$w") rc=$a"; rc=${rc:-$a}
done
mkdir -p "$OUT/verify-before"
for b in "$OUT"/bench-before/card*/bench.json; do
  [ -f "$b" ] || continue
  g=$(basename "$(dirname "$b")")
  "$PY" "$ROOT/benchmarks/pouw/harness/verify.py" "$b" --out "$OUT/verify-before/$g.json" --no-ledger > "$OUT/verify-before/$g.log" 2>&1 < /dev/null
  echo "verify $g rc=$?"
done
exit "${rc:-1}"
