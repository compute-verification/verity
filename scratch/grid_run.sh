set -eo pipefail
{ [ -L .venv ] && [ ! -e .venv ] && rm .venv; } || true
uv sync -q --all-packages --extra torch-cpu
cd verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts
for C in tied untied int8; do
  echo "=== $C grid walk"; uv run --extra torch-cpu python "$RESEARCH_RUN_DIR/inputs/grid_walk.py" "$RESEARCH_RUN_DIR/inputs/$C.program.json.gz" || echo "grid_walk rc=$?"
  echo "=== $C edges detail"; uv run --extra torch-cpu python "$RESEARCH_RUN_DIR/inputs/py_reader.py" "$RESEARCH_RUN_DIR/inputs/$C.program.json.gz" "$RESEARCH_RUN_DIR/inputs/$C.circuit.json" 2>&1 | rg -v "^\s+File|^\s{4}" || true
done
