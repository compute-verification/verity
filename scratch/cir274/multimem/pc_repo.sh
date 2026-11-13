set -u
echo "== source $(git rev-parse HEAD)"
uv sync -q --all-packages --extra torch-cpu
uv run tools/verity/check/suites.py repository --fresh --json "$OUT/suites_repository.json" --logs "$OUT/logs" 2>&1 | tail -60
echo "suites rc=${PIPESTATUS[0]}"
