uv sync -q --all-packages --extra torch-cpu </dev/null && uv run tools/verity/check/suites.py --fresh repository verity/definitions
