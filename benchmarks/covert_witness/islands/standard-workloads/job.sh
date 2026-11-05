#!/usr/bin/env bash
# trace.py on a CPU node, torch and numpy as the scorer's trace40 runs them: its pack from the plan sent with the run,
# read from the store (--store-read) and this checkout's history, then its node step on the pack:
#   research run --on vy-nebius-cpu-9 --project verity --source . --cwd source --store-read --send PLAN.json -- \
#       bash benchmarks/covert_witness/islands/standard-workloads/job.sh      (WORKERS, default 16; TRACE_TIMEOUT_S,
#       a trace's limit, default 3 h)
set -euo pipefail
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
OUT=${OUT:-$RESEARCH_RUN_DIR/out}
PLAN=$(ls "$RESEARCH_RUN_DIR"/inputs/*.json | head -1)
uv venv -q --python 3.12 "$S/venv"
uv pip install -q --python "$S/venv/bin/python" --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple "torch==2.15.0+cpu" "numpy==2.5.4" "safetensors==0.8.0" "tiktoken==0.14.0"
"$S/venv/bin/python" "$(dirname "$0")/trace.py" pack --plan "$PLAN" --out "$S/pack.tar"
TMPDIR="$S" "$S/venv/bin/python" "$(dirname "$0")/trace.py" node --pack "$S/pack.tar" --out "$OUT" --workers "${WORKERS:-16}"
