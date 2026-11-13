#!/usr/bin/env bash
# Suites at a head: the change from BASE whole (slow tests in), each suite's log and report under $OUT/logs as it ends,
# then `repository` by name (cached if the first ran it), then both long-row test files with every test's verdict.
# Usage: headsuites2.sh BASE  (run under a 16-core affinity, 64 GB)
set -o pipefail
BASE="$1"
git rev-parse HEAD | tee "$OUT/head.txt"
uv sync -q --all-packages --extra torch-cpu || exit 3
uv run --no-sync tools/verity/check/suites.py --changed "$BASE" --jobs 16 --mem-gb 64 --logs "$OUT/logs" \
  --json "$OUT/suites.json" > "$OUT/suites.txt" 2>&1
a=$?
uv run --no-sync tools/verity/check/suites.py repository --jobs 16 --mem-gb 64 --logs "$OUT/logs_repository" \
  --json "$OUT/repository.json" > "$OUT/repository.txt" 2>&1
b=$?
(cd adapters/vllm && uv run --no-sync python -m pytest -rA -p no:cacheprovider tests/pipeline/test_long_row.py \
  tests/pipeline/test_serving_rows_bits.py > "$OUT/long_row_tests.txt" 2>&1)
c=$?
echo "suites rc $a repository rc $b long_row_tests rc $c" | tee "$OUT/rc.txt"
exit $((a | b | c))
