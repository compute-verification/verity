#!/usr/bin/env bash
# A rehearsal of the sampled accept's capture with no card: Triton 3.7.1 and the pin's two kernel files in a scratch venv
# (as accept_ptx.sh), then capture_dry.py from the tree's root.  Run as: research run --on auto --cwd source
# --send capture_dry.sh --send capture_dry.py --send ptx_check.py --send ptx_sim.py --send accept_ptx.tgz
# -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/capture_dry.sh'
set -euo pipefail
PIN=d9105ea8001e0a6d77a96327d17515bb5791fb36
IN="$RESEARCH_RUN_DIR/inputs"
: "${OUT:=$RESEARCH_RUN_DIR/out}"
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
date -u +"start %Y-%m-%dT%H:%M:%SZ on $(hostname)"
uv venv -q --python 3.12 "$d/venv"
uv pip install -q --python "$d/venv/bin/python" --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
uv pip install -q --python "$d/venv/bin/python" triton==3.7.1 numpy
for f in vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py vllm/v1/worker/gpu/sample/gumbel.py; do
  mkdir -p "$d/pin/$(dirname "$f")"
  curl -fsS --retry 3 --max-time 60 -o "$d/pin/$f" "https://raw.githubusercontent.com/vllm-project/vllm/$PIN/$f"
done
tar -xzf "$IN/accept_ptx.tgz" -C "$d"
mkdir -p "$OUT/rehearsal"
TRITON_CACHE_DIR="$d/triton-cache" "$d/venv/bin/python" "$IN/capture_dry.py" --pin-src "$d/pin" --ptx "$d/accept_ptx" \
  --out "$OUT/rehearsal" "$@"
date -u +"end %Y-%m-%dT%H:%M:%SZ"
