#!/usr/bin/env bash
# trace40.py's node step on a CPU node: research run --on vy-nebius-cpu-9 --cwd source --send PACK.tar --
#   bash benchmarks/covert_witness/scorer/job40.sh   (WORKERS, default 48). torch and numpy as the toolsmith's re-trace ran them.
# A refresh on the node (refresh-node.sh) runs it beside itself with PACK, OUT and TE_TRACE_VENV, a venv it keeps.
set -euo pipefail
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
OUT=${OUT:-$RESEARCH_RUN_DIR/out}
PACK=${PACK:-$(ls "$RESEARCH_RUN_DIR"/inputs/*.tar | head -1)}
V=${TE_TRACE_VENV:-$S/venv}
if [ ! -x "$V/bin/python" ]; then
  uv venv -q --python 3.12 "$V"
  uv pip install -q --python "$V/bin/python" --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple "torch==2.15.0+cpu" "numpy==2.5.4" "safetensors==0.8.0" "tiktoken==0.14.0"
fi
TMPDIR="$S" "$V/bin/python" "$(dirname "$0")/trace40.py" node --pack "$PACK" --out "$OUT" --workers "${WORKERS:-48}"
