set -eo pipefail
{ [ -L .venv ] && [ ! -e .venv ] && rm .venv; } || true
uv sync -q --all-packages --extra torch-cpu
cd verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts
for C in tied untied int8; do echo "=== $C"; uv run --extra torch-cpu python "$RESEARCH_RUN_DIR/inputs/recut_count.py" "$RESEARCH_RUN_DIR/inputs/$C.program.json.gz" || echo "rc=$?"; done
