#!/usr/bin/env bash
# The accept kernels' PTX at the pin with no card: Triton 3.7.1 (torch 2.13.0's pin of it) and the pin's two kernel files in a
# scratch venv, compiled for sm_90 by accept_ptx.py.  Run as: research run --on auto --cwd source --send accept_ptx.sh
# --send accept_ptx.py -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/accept_ptx.sh'
set -euo pipefail
PIN=d9105ea8001e0a6d77a96327d17515bb5791fb36
IN="$RESEARCH_RUN_DIR/inputs"
: "${OUT:=$RESEARCH_RUN_DIR/out}"
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
date -u +"start %Y-%m-%dT%H:%M:%SZ on $(hostname)"
uv venv -q --python 3.12 "$d/venv"
uv pip install -q --python "$d/venv/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
uv pip install -q --python "$d/venv/bin/python" triton==3.7.1
uv pip list --python "$d/venv/bin/python" 2>/dev/null | grep -Ei '^(torch|triton|numpy) ' || true
for f in vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py vllm/v1/worker/gpu/sample/gumbel.py; do
  mkdir -p "$d/pin/$(dirname "$f")"
  curl -fsS --retry 3 --max-time 60 -o "$d/pin/$f" "https://raw.githubusercontent.com/vllm-project/vllm/$PIN/$f"
done
mkdir -p "$OUT/accept_ptx"
ls "$d/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/" > "$OUT/accept_ptx/triton_bin.txt" 2>&1 || true
TRITON_CACHE_DIR="$d/triton-cache" "$d/venv/bin/python" "$IN/accept_ptx.py" --pin-src "$d/pin" --pin "$PIN" \
  --out "$OUT/accept_ptx" "$@"
date -u +"end %Y-%m-%dT%H:%M:%SZ"
