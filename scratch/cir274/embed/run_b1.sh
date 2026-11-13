set -u
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD | tee $OUT/head.txt
sha256sum $I/multimem-join.diff | tee $OUT/diff_sha.txt
git apply $I/multimem-join.diff && git diff --stat | tee $OUT/applied.txt
uv sync -q --all-packages --extra torch-cpu
bash $I/watch.sh census4 .venv/bin/python $I/embed_census.py $OUT 4 4 > $OUT/stdout_census4.txt 2>&1; tail -4 $OUT/stdout_census4.txt | cut -c1-3000
bash $I/watch.sh census5 .venv/bin/python $I/embed_census.py $OUT 4 5 > $OUT/stdout_census5.txt 2>&1; tail -4 $OUT/stdout_census5.txt | cut -c1-3000
PY=$PWD/.venv/bin/python
cd adapters/vllm
bash $I/watch.sh join $PY -m pytest -q -p no:cacheprovider --runxfail tests/program/predict/test_node_window_join.py > $OUT/tests_join_patched.txt 2>&1; tail -3 $OUT/tests_join_patched.txt
bash $I/watch.sh tests $PY -m pytest -q -p no:cacheprovider tests/program/predict/test_node_window_embed.py tests/program/predict/test_node_window_multimem.py tests/program/predict/test_node_window_sparse.py > $OUT/tests_patched.txt 2>&1; tail -5 $OUT/tests_patched.txt
