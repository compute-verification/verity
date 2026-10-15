#!/usr/bin/env bash
# VLLM-14 scratch: py-spy profile of rank 0's GLM-5.3 fold (prefill + the first decode steps), from a shipped capture log.
set -euo pipefail
LOG=${FOLD_LOG:?}
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
PY=${PY:-/workspace/jobs/venv312/bin/python}
"$PY" -m pip install -q --target "$d/spy" py-spy >/dev/null 2>&1
export PYTHONPATH="$PWD:$PWD/../..:$PWD/../../tools/research/src"
"$d/spy/bin/py-spy" record --duration 1500 --rate 20 --format speedscope -o "$OUT/fold.speedscope.json" --nonblocking -- \
  "$PY" -m verity_vllm.pipeline.cli match --worker fold --log "$LOG" --profile derived_GLM53_tp8 --out "$d/fold_summary.json" || true
"$d/spy/bin/py-spy" record --help >/dev/null 2>&1 || true
