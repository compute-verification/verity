uv sync -q --all-packages --extra torch-cpu </dev/null && uv run tools/verity/check/suites.py --fresh repository verity/definitions verity-bench benchmarks/one_stage verity-flock
