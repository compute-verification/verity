set -u
I=$RESEARCH_RUN_DIR/inputs
echo "== source $(git rev-parse HEAD)"
git apply "$I/fix-p01-port-rows-public.diff" && git apply "$I/fix-p03-value-check-draws.diff" && echo "applied both"
git diff --stat
uv sync -q --all-packages --extra torch-cpu
PY=$PWD/.venv/bin/python
echo "== ratchet gate"
python3 adapters/vllm/tools/ratchet_gate.py b3902993caeaef278eb966882ae5c50056d89032 HEAD --json > "$OUT/ratchet.json"; echo "ratchet rc=$?"
python3 -c "import json,sys; print('failures:', json.load(open(sys.argv[1]))['failures'])" "$OUT/ratchet.json"
echo "== suites"
uv run tools/verity/check/suites.py verity-vllm --fresh --json "$OUT/suites_vllm_lint.json" --logs "$OUT/logs_lint" -- tests/lint -rfEs 2>&1 | rg -v "^\s+(delete|tests/|verity_vllm/)" | tail -25
echo "lint suites rc=${PIPESTATUS[0]}"
uv run tools/verity/check/suites.py verity/commitments/windows repository --fresh --json "$OUT/suites_core_repo.json" --logs "$OUT/logs_core" 2>&1 | tail -25
echo "core+repo suites rc=${PIPESTATUS[0]}"
echo "== value check, 1 layer, patched API"
cd adapters/vllm
bash $I/watch.sh values1 $PY -u $I/values_run_p3.py $OUT --ids 32 --layers 1 --name p3_1l_32 --no-plants > $OUT/stdout_values.txt 2>&1
echo "values rc=$?"
rg "check:|window |wrote|Error|Traceback" $OUT/stdout_values.txt | cut -c1-600 | tail -8
