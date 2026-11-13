set -u
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD | tee $OUT/head.txt; git status --short | tee $OUT/status.txt
uv sync -q --all-packages --extra torch-cpu
PY=$PWD/.venv/bin/python
cd adapters/vllm
bash $I/watch.sh tests $PY -m pytest -q -p no:cacheprovider -rxXs tests/program/predict/test_node_window_embed.py tests/program/predict/test_node_window_join.py tests/program/predict/test_node_window_multimem.py tests/program/predict/test_node_window_sparse.py > $OUT/tests.txt 2>&1
rc=$?; tail -8 $OUT/tests.txt; exit $rc
