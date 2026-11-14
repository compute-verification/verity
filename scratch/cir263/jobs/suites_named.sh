#!/usr/bin/env bash
# CIR-263: the suites named in SUITES (space-separated), each run by name, to $OUT.  Leaves TMPDIR as the run gives it.
set -uo pipefail
: "${SUITES:?SUITES, the suites to run}"
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
date -u +"suites %FT%TZ"
# shellcheck disable=SC2086
uv run tools/verity/check/suites.py $SUITES --mem-gb 32 --json "$OUT/suites.json" --logs "$OUT/logs" > "$OUT/suites.log" 2>&1
s=$?
echo "suites rc=$s"; grep -vE "^ +ran because" "$OUT/suites.log" | tail -30
date -u +"end %FT%TZ"
exit $s
