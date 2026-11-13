uv sync -q --all-packages --extra torch-cpu </dev/null && uv run tools/verity/check/suites.py --fresh --mem-gb 48 verity-vllm
