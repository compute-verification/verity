#!/usr/bin/env bash
# one suite by name (SUITE), whole, with its verdict in $OUT/suites.json
set -u
uv sync -q --all-packages --extra torch-cpu >"$OUT/sync.log" 2>&1 || { echo "uv sync failed"; tail -20 "$OUT/sync.log"; exit 2; }
start=$(date -u +%FT%TZ)
uv run tools/verity/check/suites.py "${SUITE:?}" --json "$OUT/suites.json" --logs "$OUT/logs" >"$OUT/suites.txt" 2>&1
rc=$?
echo "start $start end $(date -u +%FT%TZ) rc $rc" >"$OUT/rc.txt"
tail -8 "$OUT/suites.txt" | cut -c1-300
exit $rc
