#!/usr/bin/env bash
# After a merge of main: the region's test file, the size record rewritten by its own tool (main's tool keeps an entry
# whose description has not moved), its test, and STATEMENTS' test.  Usage: merge2.sh
set -o pipefail
run="$(basename "$RESEARCH_RUN_DIR")"
R=adapters/vllm/tools/description_sizes.json
git rev-parse HEAD | tee "$OUT/head.txt"
uv sync -q --all-packages --extra torch-cpu || exit 3
uv run --no-sync python -m pytest -rA -p no:cacheprovider -o verbosity_assertions=2 \
  verity/definitions/tests/test_dsa_topk_reads.py > "$OUT/region_tests.txt" 2>&1
a=$?
cp "$R" "$OUT/committed_record.json"
uv run --no-sync python adapters/vllm/tools/description_sizes.py update -j 8 --run "$run" > "$OUT/update.txt" 2>&1
u=$?
cp "$R" "$OUT/description_sizes.json"
git diff --stat -- "$R" > "$OUT/record_diff.txt"; git diff -- "$R" >> "$OUT/record_diff.txt"
(cd adapters/vllm && uv run --no-sync python -m pytest -q -rA -p no:cacheprovider tests/tools/test_description_sizes.py > "$OUT/sizes_tests.txt" 2>&1)
b=$?
(cd adapters/vllm && uv run --no-sync python -m pytest -q -rA -p no:cacheprovider -o verbosity_assertions=2 \
  tests/pipeline/test_served_unit_rows.py > "$OUT/statements.txt" 2>&1)
c=$?
echo "region rc $a update rc $u sizes rc $b statements rc $c" | tee "$OUT/rc.txt"
exit $((a | u | b | c))
