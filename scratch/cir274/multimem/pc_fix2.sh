set -u
I=$RESEARCH_RUN_DIR/inputs
echo "== source $(git rev-parse HEAD)"
git apply "$I/fix-p01-port-rows-public.diff" && git apply "$I/fix-p03-value-check-draws.diff" && echo "applied both"
git diff --stat
sha256sum "$I"/fix-p0*.diff
uv sync -q --all-packages --extra torch-cpu
echo "== ratchet gate"
python3 adapters/vllm/tools/ratchet_gate.py b3902993caeaef278eb966882ae5c50056d89032 HEAD --json > "$OUT/ratchet.json"; echo "ratchet rc=$?"
python3 -c "import json,sys; print('failures:', json.load(open(sys.argv[1]))['failures'])" "$OUT/ratchet.json"
echo "== suites"
uv run tools/verity/check/suites.py verity-vllm --fresh --json "$OUT/suites_vllm_lint.json" --logs "$OUT/logs_lint" -- tests/lint -rfEs 2>&1 | rg -v "^\s+(delete|tests/|verity_vllm/)" | tail -25
echo "lint suites rc=${PIPESTATUS[0]}"
