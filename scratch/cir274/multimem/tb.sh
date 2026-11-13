set -u
I=$RESEARCH_RUN_DIR/inputs
BASE=b3902993caeaef278eb966882ae5c50056d89032
echo "== source $(git rev-parse HEAD 2>&1)"
uv sync -q --all-packages --extra torch-cpu
PY=$PWD/.venv/bin/python
echo "== ratchet gate"
git cat-file -e "$BASE^{commit}" 2>/dev/null || git fetch -q origin "$BASE"
python3 adapters/vllm/tools/ratchet_gate.py "$BASE" HEAD --json > "$OUT/ratchet.json"
echo "ratchet rc=$?"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('failures:', d['failures'])" "$OUT/ratchet.json"
echo "== census 4"
bash $I/watch.sh census4 $PY $I/mm_census.py $OUT 4 2>&1 | tail -30
cd adapters/vllm
echo "== pytest"
bash $I/watch.sh tests $PY -m pytest -q -p no:cacheprovider -n 4 --junitxml "$OUT/junit.xml" \
  tests/program/predict/test_node_window_multimem.py tests/program/predict/test_node_window.py \
  tests/program/predict/test_node_window_dense.py tests/program/predict/test_node_window_sparse.py 2>&1 | tail -30
echo "== new test alone"
$PY -m pytest -q -p no:cacheprovider --junitxml "$OUT/junit_new.xml" \
  "tests/program/predict/test_node_window_sparse.py::test_chain_place_ranges" 2>&1 | tail -5
