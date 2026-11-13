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
cd adapters/vllm
echo "== pytest"
bash $I/watch.sh tests $PY -m pytest -q -p no:cacheprovider -n 4 --junitxml "$OUT/junit.xml" \
  tests/pipeline/test_tp_wires.py tests/program/test_tp_binding.py \
  tests/commit/test_memory_map.py tests/pipeline/test_tp_build_relabel.py tests/program/test_prune.py \
  tests/program/predict/test_tp_steps.py tests/program/predict/test_steps.py tests/program/test_batched.py \
  2>&1 | tail -40
echo "== mypy"
$PY -m mypy --config-file pyproject.toml --cache-dir /dev/null --follow-imports skip \
  verity_vllm/pipeline/tp/wires.py verity_vllm/program/prune.py verity_vllm/program/tp_binding.py \
  tests/pipeline/test_tp_wires.py tests/program/test_tp_binding.py 2>&1 | tee "$OUT/mypy.txt"
echo "mypy rc=${PIPESTATUS[0]}"
