# the patched tree: the sparse test file, then the window across prompt lengths
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD && git apply $I/fix-chain-ranges.diff && git diff --stat
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 9
(cd adapters/vllm && timeout 3000 ../../.venv/bin/python -m pytest -q -x tests/program/predict/test_node_window_sparse.py -p no:cacheprovider 2>&1 | tail -5); echo pytest rc=$?
.venv/bin/python $I/p3_fixcheck.py $OUT 2 43 64 65 86 127 128 129 150 200 256; echo fix2 rc=$?
.venv/bin/python $I/p3_fixcheck.py $OUT 3 43 127 128 150; echo fix3 rc=$?
