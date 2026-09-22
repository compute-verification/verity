#!/usr/bin/env bash
# [REPLAY, additive] Pod bootstrap for the trusted-replay lane (GUIDANCE §32.9A/.10).
# Environment = TA1's A100 environment exactly (env/README.md cu129 deviation table):
# pinned vLLM d9105ea80 cu129 wheel (sha 7aa52ac7...), torch 2.13.0+cu129, SmolLM2-135M
# snapshot @ 93efa2f0 with sha-verified safetensors. Idempotent.
set -euxo pipefail

WHEEL_URL="https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
WHEEL_FILE="vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
WHEEL_SHA="7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3"
REVISION="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

if [ ! -x /workspace/venv/bin/python ]; then
  python3 -m venv /workspace/venv
  /workspace/venv/bin/pip install -U pip uv
fi

mkdir -p /workspace/wheels && cd /workspace/wheels
if [ ! -f "$WHEEL_FILE" ]; then
  curl -fLo "$WHEEL_FILE" "$WHEEL_URL"
fi
echo "$WHEEL_SHA  $WHEEL_FILE" | sha256sum -c

/workspace/venv/bin/python -c "import vllm" 2>/dev/null || \
/workspace/venv/bin/uv pip install --python /workspace/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
  "/workspace/wheels/$WHEEL_FILE" numpy safetensors huggingface_hub zstandard

export HF_HOME=/workspace/hf
SNAP="/workspace/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/$REVISION"
if [ ! -f "$SNAP/model.safetensors" ]; then
  /workspace/venv/bin/python - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download("HuggingFaceTB/SmolLM2-135M", revision="$REVISION")
print(p)
PYEOF
fi
echo "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  $SNAP/model.safetensors" | sha256sum -c
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
echo SETUP-OK
