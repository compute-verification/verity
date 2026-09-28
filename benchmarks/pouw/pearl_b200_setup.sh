#!/usr/bin/env bash
# Pearl's own vLLM miner (pearl-research-labs/pearl PR #311 head) on a B200 pod: uv, Rust (for py-pearl-mining), the
# checkout at the pinned commit, `uv sync --package vllm-miner` (CUDA 13 torch, vLLM 0.28), and the calibration model.
# Idempotent; writes /root/pearl-setup.done when everything is in place.  Run it detached: setsid nohup bash pearl_b200_setup.sh &
set -euo pipefail
PEARL_SHA=${PEARL_SHA:-284b147bf56bcc7e38c9f0bb87a56d61e9b7519d}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
command -v git >/dev/null && dpkg -s libatomic1 >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git libatomic1 curl >/dev/null; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
if [ ! -d /root/pearl/.git ]; then
  mkdir -p /root/pearl && cd /root/pearl && git init -q && git remote add origin https://github.com/pearl-research-labs/pearl.git
  git fetch -q --depth 1 origin "$PEARL_SHA" && git checkout -q FETCH_HEAD
fi
cd /root/pearl
export UV_TORCH_BACKEND=cu130 UV_PROJECT_ENVIRONMENT=/root/pearl-venv UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/root/uv-python
uv python install 3.12
uv sync --package vllm-miner --no-dev
/root/pearl-venv/bin/python -c "import vllm, torch, vllm_miner; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
/root/pearl-venv/bin/hf download "$MODEL" --local-dir "/root/models/$(basename "$MODEL")" > /root/hf.log 2>&1
git -C /root/pearl rev-parse HEAD > /root/pearl-setup.done
