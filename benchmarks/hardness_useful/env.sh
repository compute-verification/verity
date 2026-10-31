# Source me: the CUDA torch venv, the token cache and the models' cache under $HU_BASE, built once (vy-nebius-1, RTX PRO 6000).
HU_BASE=${HU_BASE:-/workspace/research/lanes/compute-research/hardness-useful}
mkdir -p "$HU_BASE"
(
  flock 9
  if [ ! -x "$HU_BASE/venv/bin/python" ] || [ ! -e "$HU_BASE/venv/.complete" ]; then
    rm -rf "$HU_BASE/venv"
    uv venv -q --python 3.12 "$HU_BASE/venv"
    uv pip install -q --python "$HU_BASE/venv/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    uv pip install -q --python "$HU_BASE/venv/bin/python" numpy tokenizers huggingface_hub hf_xet pyarrow transformers \
      safetensors datasets "lm_eval>=0.4.8"
    touch "$HU_BASE/venv/.complete"
  fi
) 9>"$HU_BASE/venv.lock"
export PATH=$HU_BASE/venv/bin:$PATH
export HF_HOME=${HF_HOME:-/workspace/hf}
export TORCHINDUCTOR_CACHE_DIR=$HU_BASE/inductor-cache TRITON_CACHE_DIR=$HU_BASE/triton-cache
export HU_BASE HU_DATA=$HU_BASE/fineweb-edu-llama3
