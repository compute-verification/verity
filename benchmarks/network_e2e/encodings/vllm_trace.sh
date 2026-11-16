#!/usr/bin/env bash
# vllm_trace.py's traces and the replay's check of each, on a CPU, under the integration's pinned vLLM (the wheel
# pod_bootstrap.sh names): the node's venv where it holds that vLLM, else one built in a scratch directory and removed.
#   research run --on auto --cwd source -- bash benchmarks/network_e2e/encodings/vllm_trace.sh [vllm_trace.py's arguments]
# Writes OUT/traces/<case>.jsonl (and <case>.tokens.json with prefix caching), OUT/traces/check.json and OUT/versions.txt;
# exits 1 unless every replay's steps are its trace's. VENV: a venv to try first.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:?set OUT}
BOOT=$ROOT/adapters/vllm/verity_vllm/ops/pod_bootstrap.sh
eval "$(grep -E '^(W|WSHA|TORCHCODEC)=' "$BOOT")"
PIN=${W#*+g}
PIN=${PIN%%.*}
mkdir -p "$OUT"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
PY=""
for v in "${VENV:-}" /workspace/venv312 /workspace/cp/venv312; do
  if [ -n "$v" ] && [ -x "$v/bin/python" ] && "$v/bin/python" -c "import vllm" 2>/dev/null \
      && "$v/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null | grep -q "$PIN"; then
    PY=$v/bin/python
    break
  fi
done
if [ -z "$PY" ]; then
  URL=$(grep -o 'https://wheels.vllm.ai/[^"]*' "$BOOT")
  export UV_CACHE_DIR=$scratch/uv UV_HTTP_TIMEOUT=600
  uv venv "$scratch/venv" --python 3.12 > /dev/null
  curl -fsSLo "$scratch/$W" "$URL"
  echo "$WSHA  $scratch/$W" | sha256sum -c > /dev/null
  uv pip install --python "$scratch/venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match "$scratch/$W" "$TORCHCODEC" "numpy>=1.26" > "$OUT/pip.log" 2>&1
  PY=$scratch/venv/bin/python
fi
export PYTHONPATH=$ROOT:$ROOT/adapters/vllm VLLM_TARGET_DEVICE=cpu HF_HOME=${HF_HOME:-$scratch/hf}
"$PY" -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)" > "$OUT/versions.txt"
"$PY" -P "$HERE/vllm_trace.py" "$OUT/traces" --check "$@"
