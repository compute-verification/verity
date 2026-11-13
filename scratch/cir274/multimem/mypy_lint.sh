set -u
echo "== source $(git rev-parse HEAD)"
uv sync -q --all-packages --extra torch-cpu
uv run tools/verity/check/suites.py verity-vllm --fresh --json "$OUT/suites_mypy.json" --logs "$OUT/logs" -- tests/lint/test_mypy.py -rfEs 2>&1 | rg -v "^\s+(delete|warnings\.warn)" | tail -20
echo "suites rc=${PIPESTATUS[0]}"
