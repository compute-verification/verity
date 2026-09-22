#!/usr/bin/env bash
# [PAD-INT, additive] Pod bootstrap for lane PAD-INT on the shared `derive-dev` RTX 4090 pod (GUIDANCE §44.2/§44.3).
#
# Own prefix /workspace/padint-env — never touches DERIVE's /workspace/venv or CAPCORR's /workspace/capcorr-env.
#   /workspace/padint-env/venv-cu129   the pinned vLLM interpreter of every B0 capture so far
#                                      (vLLM d9105ea80 cu129 wheel, sha 7aa52ac7…; torch 2.13.0+cu129) — exactly
#                                      zk/campaign/intpad2_vllm_env.sh, relocated
#   /workspace/padint-env/hf           HF_HOME with SmolLM2-135M @ 93efa2f0 (model.safetensors sha 80521b40… verified)
#   /workspace/veritor-padint/.venv    the driver venv (uv sync; Python 3.12 with numpy — the challenge core sampler
#                                      needs it, VERITY_CORE_PYTHON)
#   /workspace/padint-env/veritor-zk-host  the certified record's OWN shipped CUDA host binary (sha = the record
#                                      manifest's inputs.host_sha256); no cargo build
# Idempotent.  No credentials.  Usage: bash scripts/padint_pod_setup.sh  (from any cwd on the pod)
set -uo pipefail
ENV=/workspace/padint-env
REPO=${REPO:-/workspace/veritor-padint}
RECORD=${RECORD:-$REPO/fixtures/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z}
VENV=$ENV/venv-cu129
PY=$VENV/bin/python
export HF_HOME=$ENV/hf
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$ENV" /workspace/padint-results /workspace/padint-cache
stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }

stage machine
nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || true

stage "vLLM venv (pinned wheel)"
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip uv
fi
mkdir -p "$ENV/wheels" && cd "$ENV/wheels"
W="vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
[ -f "$W" ] || curl -fLo "$W" "https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
echo "7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3  $W" | sha256sum -c
"$PY" -c "import vllm" 2>/dev/null || "$VENV/bin/uv" pip install --python "$PY" \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
    "$ENV/wheels/$W" numpy safetensors huggingface_hub zstandard pytest
"$PY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

stage "checkpoint (HuggingFaceTB/SmolLM2-135M @ 93efa2f0)"
REVISION=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
SNAP="$HF_HOME/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/$REVISION"
if [ ! -f "$SNAP/model.safetensors" ]; then
  "$PY" - <<PYEOF
from huggingface_hub import snapshot_download
print(snapshot_download("HuggingFaceTB/SmolLM2-135M", revision="$REVISION",
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
PYEOF
fi
echo "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  $SNAP/model.safetensors" | sha256sum -c
ln -sfn "$SNAP" "$ENV/ckpt"

stage "driver venv (uv sync in $REPO)"
cd "$REPO"
if [ ! -x .venv/bin/python ]; then
  uv python install 3.12 >/dev/null 2>&1 || true
  uv sync --frozen >/dev/null 2>&1 || uv sync
fi
.venv/bin/python -c "import veritor, numpy, verity.ir; print('driver venv ok', numpy.__version__)"

stage "record host binary (the certified record's own CUDA host)"
test -f "$RECORD/manifest.json"
HOST_PIN=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['inputs']['host_sha256'])" "$RECORD/manifest.json")
echo "$HOST_PIN  $RECORD/bin/veritor-zk-host" | sha256sum -c
chmod +x "$RECORD/bin/veritor-zk-host"
ln -sfn "$RECORD/bin/veritor-zk-host" "$ENV/veritor-zk-host"
SP1_PROVER=cpu "$ENV/veritor-zk-host" info 2>&1 | tail -4 || echo "host info failed (check CUDA / glibc)"
echo PADINT-SETUP-OK
