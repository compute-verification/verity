#!/usr/bin/env bash
# the named suites (SUITES, space-separated; quick tier), each with its outcome record
set -uo pipefail
OUT=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/head.txt"
date -u +"start %FT%TZ"
uv sync -q --all-packages --extra torch-cpu || exit 2
date -u +"synced %FT%TZ"
rc=0
# shellcheck disable=SC2086
uv run tools/verity/check/suites.py ${SUITES:?} --quick --timeouts --json "$OUT/suites.json" --logs "$OUT/logs" \
  > "$OUT/suites.txt" 2>&1 || rc=$?
tail -n 40 "$OUT/suites.txt"
date -u +"named %FT%TZ rc=$rc"
echo "rc named=$rc" > "$OUT/rc.txt"
exit $rc
