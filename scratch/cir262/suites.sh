#!/usr/bin/env bash
# the suites the change from BASE touches (quick tier) and the repository suite whole, each with its outcome record
set -uo pipefail
OUT=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$OUT"
BASE=${BASE:?BASE: the merge base with main}
git rev-parse HEAD > "$OUT/head.txt"
date -u +"start %FT%TZ"
uv sync -q --all-packages --extra torch-cpu || exit 2
date -u +"synced %FT%TZ"
rc=0
uv run tools/verity/check/suites.py --changed "$BASE" --quick --timeouts --json "$OUT/suites.json" --logs "$OUT/logs" \
  > "$OUT/suites.txt" 2>&1 || rc=$?
tail -n 60 "$OUT/suites.txt"
date -u +"changed %FT%TZ rc=$rc"
r2=0
uv run tools/verity/check/suites.py repository --timeouts --json "$OUT/suites-repository.json" --logs "$OUT/logs-repository" \
  > "$OUT/suites-repository.txt" 2>&1 || r2=$?
tail -n 60 "$OUT/suites-repository.txt"
date -u +"repository %FT%TZ rc=$r2"
echo "rc changed=$rc repository=$r2" > "$OUT/rc.txt"
[ $rc = 0 ] && rc=$r2
exit $rc
