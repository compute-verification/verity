#!/usr/bin/env bash
# Suites at a head: the change from BASE (--quick keeps the slow tests of the test modules it reaches), then verity-vllm and
# repository by name, then both long-row test files with every test's verdict.  Usage: headsuites.sh BASE
set -o pipefail
BASE="$1"
git rev-parse HEAD | tee "$OUT/head.txt"
uv sync -q --all-packages --extra torch-cpu || exit 3
uv run --no-sync tools/verity/check/suites.py --changed "$BASE" --quick > "$OUT/suites.txt" 2>&1
a=$?
uv run --no-sync tools/verity/check/suites.py verity-vllm repository --quick > "$OUT/named.txt" 2>&1
b=$?
(cd adapters/vllm && uv run --no-sync python -m pytest -rA -p no:cacheprovider tests/pipeline/test_long_row.py \
  tests/pipeline/test_serving_rows_bits.py > "$OUT/long_row_tests.txt" 2>&1)
c=$?
echo "suites rc $a named rc $b long_row_tests rc $c" | tee "$OUT/rc.txt"
exit $((a | b | c))
