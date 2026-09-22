#!/usr/bin/env bash
# [CAPCORR, additive] Pod bootstrap for lane CAPCORR (GUIDANCE §43.5 S4/S5) on the shared `derive-dev` pod.
# Own prefix /workspace/capcorr-env (never touches DERIVE's /workspace/venv): a uv Python 3.12 venv with the
# repo's runtime deps (numpy, networkx, safetensors, huggingface_hub), the pinned SmolLM2-135M snapshot
# @ 93efa2f0 (sha-verified model.safetensors — the record's checkpoint), and the record's OWN shipped host
# binary (`<record>/bin/veritor-zk-host`, sha = manifest inputs.host_sha256) made executable for S5; no cargo
# build unless CAPCORR_BUILD_HOST=1.  Idempotent.  No credentials.
set -euo pipefail
ENV=/workspace/capcorr-env
REPO=${REPO:-/workspace/veritor-capcorr}
RECORD=${RECORD:-$REPO/fixtures/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z}
REVISION=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
mkdir -p "$ENV" /workspace/capcorr
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }

stage machine
nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true

stage "uv + venv"
command -v uv >/dev/null || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null
uv python install 3.12 >/dev/null 2>&1 || true
[ -x "$ENV/venv/bin/python" ] || uv venv --python 3.12 "$ENV/venv" >/dev/null
"$ENV/venv/bin/python" -c "import numpy, networkx, safetensors, huggingface_hub" 2>/dev/null || \
  uv pip install --python "$ENV/venv/bin/python" "numpy>=2.3" "networkx>=3.6" safetensors huggingface_hub >/dev/null
"$ENV/venv/bin/python" -c "import numpy, networkx, safetensors; print('py deps ok', numpy.__version__)"

stage "checkpoint (HuggingFaceTB/SmolLM2-135M @ $REVISION)"
export HF_HOME=$ENV/hf
SNAP="$HF_HOME/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/$REVISION"
if [ ! -f "$SNAP/model.safetensors" ]; then
  "$ENV/venv/bin/python" - <<PYEOF
from huggingface_hub import snapshot_download
print(snapshot_download("HuggingFaceTB/SmolLM2-135M", revision="$REVISION", allow_patterns=["*.json", "*.safetensors"]))
PYEOF
fi
echo "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  $SNAP/model.safetensors" | sha256sum -c
ln -sfn "$SNAP" "$ENV/ckpt"

stage "record + host binary"
test -f "$RECORD/manifest.json"
HOST_PIN=$("$ENV/venv/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1]))['inputs']['host_sha256'])" "$RECORD/manifest.json")
echo "$HOST_PIN  $RECORD/bin/veritor-zk-host" | sha256sum -c
chmod +x "$RECORD/bin/veritor-zk-host"
ln -sfn "$RECORD/bin/veritor-zk-host" "$ENV/veritor-zk-host"
"$ENV/veritor-zk-host" info 2>&1 | tail -8 || echo "host info failed (check CUDA / glibc)"

if [ "${CAPCORR_BUILD_HOST:-0}" = "1" ]; then
  stage "build host from zk/sp1 (cuda)"
  command -v cargo >/dev/null || (curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null)
  cargo prove --version 2>/dev/null | grep -q f66b4bf || { curl -fsSL https://sp1up.succinct.xyz | bash >/dev/null; "$HOME/.sp1/bin/sp1up" --version v6.4.0; }
  (cd "$REPO/zk/sp1" && CARGO_TARGET_DIR=$ENV/target cargo build --release -p veritor-zk-host --features cuda 2>&1 | tail -3)
  cp "$ENV/target/release/veritor-zk-host" "$ENV/veritor-zk-host-built"
  sha256sum "$ENV/veritor-zk-host-built"
fi
echo CAPCORR-SETUP-OK
