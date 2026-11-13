set -u
I=$RESEARCH_RUN_DIR/inputs
uv sync -q --all-packages --extra torch-cpu
PY=$PWD/.venv/bin/python
cd adapters/vllm
bash $I/watch.sh tests $PY -m pytest -q -p no:cacheprovider -n 4 tests/program/predict/test_node_window_multimem.py tests/program/predict/test_node_window.py tests/program/predict/test_node_window_dense.py tests/program/predict/test_node_window_sparse.py 2>&1 | tail -30
echo "== mypy"
$PY -m mypy --config-file pyproject.toml --cache-dir /dev/null --follow-imports skip verity_vllm/program/predict/node_window_multimem.py tests/program/predict/test_node_window_multimem.py
