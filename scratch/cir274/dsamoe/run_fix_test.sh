# the chain-range test: it fails without the fix and passes with it (the whole sparse test file, timed)
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 9
git apply $I/test-only.diff && git diff --stat
/usr/bin/time -f "%e s %M KB" .venv/bin/python -m pytest -q -p no:cacheprovider adapters/vllm/tests/program/predict/test_node_window_sparse.py -k chain_place_ranges 2>&1 | tail -6; echo "without the fix: rc=${PIPESTATUS[0]}"
git checkout -- . && git apply $I/fix-chain-ranges-with-test.diff && git diff --stat
/usr/bin/time -f "%e s %M KB" .venv/bin/python -m pytest -q -p no:cacheprovider adapters/vllm/tests/program/predict/test_node_window_sparse.py 2>&1 | tail -6; echo "with the fix: rc=${PIPESTATUS[0]}"
