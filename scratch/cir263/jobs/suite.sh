#!/usr/bin/env bash
# One suite whole, slow tests included, to $OUT/suites.json: bash suite.sh NAME
set -uo pipefail
name=${1:?usage: suite.sh NAME}
date -u +"sync %FT%TZ"
uv sync -q --all-packages --extra torch-cpu || exit 2
date -u +"$name %FT%TZ"
uv run tools/verity/check/suites.py "$name" --json "$OUT/suites.json" --logs "$OUT/logs" > "$OUT/suites.log" 2>&1
rc=$?
echo "$name rc=$rc"
tail -30 "$OUT/suites.log"
date -u +"end %FT%TZ"
exit $rc
