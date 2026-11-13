set -u
echo "== source $(git rev-parse HEAD)"
uv sync -q --all-packages --extra torch-cpu
uv run tools/verity/check/suites.py verity-vllm --fresh --json "$OUT/suites_vllm_lint.json" --logs "$OUT/logs" -- tests/lint -rfEs 2>&1 | tail -80
echo "suites rc=${PIPESTATUS[0]}"
