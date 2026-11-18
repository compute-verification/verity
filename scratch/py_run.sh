set -eo pipefail
CLASS=$1; SRC=$2
{ [ -L .venv ] && [ ! -e .venv ] && rm .venv; } || true
uv sync -q --all-packages --extra torch-cpu
git log -1 --format='source %h' 2>/dev/null || true
cd verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts
S=$(date +%s)
/usr/bin/time -v uv run --extra torch-cpu python "$RESEARCH_RUN_DIR/inputs/py_reader.py" "$SRC/$CLASS.program.json.gz" "$SRC/$CLASS.circuit.json" 2> "$OUT/time.txt" || echo "READER rc=$?"
rg -n "Maximum resident|Elapsed" "$OUT/time.txt" || true
echo "PY_DONE in $(( $(date +%s) - S )) s"
