#!/bin/bash
# W3c pod (946fc35f6a06): driver 570.158.01 (CUDA 12.8) cannot run the pinned torch 2.13.0+cu130
# (needs driver >= 580).  Install the SAME vLLM commit (d9105ea80) from its cu129 wheel variant into a
# second venv, with torch 2.13.0+cu129; every other pin comes from env/requirements.lock.  CUDA 12.x
# minor-version compatibility lets a cu129 runtime run on a 12.8 driver.  Deviation recorded in the
# experiment README (env.torch = 2.13.0+cu129, env.vllm_wheel = ...cu129...).
set -u
export PIP_DISABLE_PIP_VERSION_CHECK=1
VENV=/workspace/venv-cu129
W=vllm-0.28.1rc1.dev472+gd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl
URL="https://wheels.vllm.ai/d9105ea8001e0a6d77a96327d17515bb5791fb36/vllm-0.28.1rc1.dev472%2Bgd9105ea80.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
mkdir -p /workspace/wheels /workspace/logs
cd /workspace/wheels
if [ ! -f "$W" ]; then curl -fL -o "$W" "$URL" || exit 1; fi
sha256sum "$W" | tee /workspace/wheels/$W.sha256
[ -d $VENV ] || python3 -m venv $VENV || exit 1
$VENV/bin/pip install -q -U pip uv || exit 1
# torch/vision/audio from the cu129 index; triton comes with torch; drop the cu13 nvidia libs.
grep -vE '^(vllm|torch|torchvision|torchaudio|torchcodec|torch_c_dlpack_ext|nvidia-|triton|cuda-|xformers|flashinfer)' \
  /workspace/veritor/vllm-poc/env/requirements.lock > /workspace/wheels/requirements.cu129.rest
$VENV/bin/uv pip install --python $VENV/bin/python \
  "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" --torch-backend=cu129 || exit 1
$VENV/bin/uv pip install --python $VENV/bin/python "/workspace/wheels/$W" --torch-backend=cu129 || exit 1
$VENV/bin/uv pip install --python $VENV/bin/python -r /workspace/wheels/requirements.cu129.rest --torch-backend=cu129 || exit 1
$VENV/bin/python - <<'EOF'
import torch, vllm, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("vllm", vllm.__version__)
torch.cuda.init()
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
print("cuda ok", torch.cuda.get_device_name(), (x @ x).sum().item())
EOF
$VENV/bin/pip freeze > /workspace/wheels/requirements.cu129.freeze
echo BOOTSTRAP_CU129_DONE
