#!/usr/bin/env bash
# CIR-263 before ready: every suite the change from its base touches, quick (the test modules it reaches keep their slow tests),
# to $OUT/suites.json; then the root `repository` suite whole, slow tests included, to $OUT/suites-repository.json.
set -uo pipefail
BASE=3498d4ad6546bed9289878c1b4cf19f218d7478e
date -u +"sync %FT%TZ"
uv sync -q --all-packages --extra torch-cpu || exit 2
date -u +"changed %FT%TZ"
uv run tools/verity/check/suites.py --mem-gb 32 --changed "$BASE" --quick --json "$OUT/suites.json" --logs "$OUT/logs" > "$OUT/suites.log" 2>&1
a=$?
echo "changed rc=$a"
tail -80 "$OUT/suites.log"
date -u +"repository %FT%TZ"
uv run tools/verity/check/suites.py repository --json "$OUT/suites-repository.json" --logs "$OUT/logs-repository" > "$OUT/suites-repository.log" 2>&1
b=$?
echo "repository rc=$b"
tail -30 "$OUT/suites-repository.log"
date -u +"end %FT%TZ"
[ "$a" -eq 0 ] && [ "$b" -eq 0 ]
