#!/usr/bin/env bash
# trace40.py's node step on a CPU node: research run --on vy-nebius-cpu-9 --cwd source --send PACK.tar --
#   bash benchmarks/covert_witness/scorer/job40.sh   (WORKERS, default 48). torch and numpy as the toolsmith's re-trace ran them.
set -euo pipefail
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
OUT=${OUT:-$RESEARCH_RUN_DIR/out}
PACK=$(ls "$RESEARCH_RUN_DIR"/inputs/*.tar | head -1)
uv venv -q --python 3.12 "$S/venv"
uv pip install -q --python "$S/venv/bin/python" --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple "torch==2.15.0+cpu" "numpy==2.5.4" "safetensors==0.8.0" "tiktoken==0.14.0"
TMPDIR="$S" "$S/venv/bin/python" "$(dirname "$0")/trace40.py" node --pack "$PACK" --out "$OUT" --workers "${WORKERS:-48}"
