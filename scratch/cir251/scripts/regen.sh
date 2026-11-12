set -o pipefail
run="$(basename "$RESEARCH_RUN_DIR")"
R=adapters/vllm/tools/description_sizes.json
git rev-parse HEAD | tee "$OUT/head.txt"
uv sync -q --all-packages --extra torch-cpu || exit 3
cp "$R" "$OUT/main_record.json"
uv run --no-sync python adapters/vllm/tools/description_sizes.py update -j 8 --run "$run" 2>&1 | tee "$OUT/update.txt" || exit 4
cp "$R" "$OUT/updated_record.json"
uv run --no-sync python "$RESEARCH_RUN_DIR/inputs/keep.py" "$OUT/main_record.json" 2>&1 | tee "$OUT/keep.json" || exit 5
cp "$R" "$OUT/description_sizes.json"
(cd adapters/vllm && uv run --no-sync python -m pytest -q -rA -p no:cacheprovider tests/tools/test_description_sizes.py 2>&1 | tee "$OUT/pytest.txt")
a=$?
b=0
if [ "${1:-}" = statements ]; then
  (cd adapters/vllm && uv run --no-sync python -m pytest -q -rA -p no:cacheprovider tests/pipeline/test_served_unit_rows.py 2>&1 | tee "$OUT/statements.txt")
  b=$?
fi
echo "sizes rc $a statements rc $b" | tee "$OUT/rc.txt"
exit $a
