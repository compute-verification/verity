#!/usr/bin/env bash
# Which clause of the reference's `nodes_held` refuses `thursday_case.py`'s declared case at LAYERS, class R,P,O.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
uv sync -q --all-packages --extra torch-cpu || exit 2
NO_REFERENCE=1 .venv/bin/python -u benchmarks/inference_only/own_rows/thursday_case.py "$PWD" "$OUT" "${1:?LAYERS}" "${2:-2,9,2}" 2>&1 | cut -c1-600
.venv/bin/python -u benchmarks/inference_only/own_rows/why_nodes.py "$OUT/thursday-nodes.case.json" 2>&1 | cut -c1-900
