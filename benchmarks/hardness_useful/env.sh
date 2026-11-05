HU_BASE=${HU_BASE:-/workspace/research/lanes/compute-research/hardness-useful}
mkdir -p "$HU_BASE"
HU_PKGS="numpy tokenizers huggingface_hub hf_xet pyarrow transformers==5.22.1 safetensors datasets accelerate lm_eval>=0.4.8"
HU_TORCH=cu130
command -v nvidia-smi >/dev/null || HU_TORCH=cpu
( flock 9
  if [ ! -x "$HU_BASE/venv/bin/python" ]; then
    uv venv -q --python 3.12 "$HU_BASE/venv"
    uv pip install -q --python "$HU_BASE/venv/bin/python" torch==2.14.1 --index-url "https://download.pytorch.org/whl/$HU_TORCH"
  fi
  if [ "$(cat "$HU_BASE/venv/.complete" 2>/dev/null)" != "$HU_PKGS" ]; then
    uv pip install -q --python "$HU_BASE/venv/bin/python" $HU_PKGS
    echo "$HU_PKGS" > "$HU_BASE/venv/.complete"
  fi ) 9>"$HU_BASE/venv.lock"
export PATH=$HU_BASE/venv/bin:$PATH
export HF_HOME=${HF_HOME:-/workspace/hf}
export TORCHINDUCTOR_CACHE_DIR=$HU_BASE/inductor-cache TRITON_CACHE_DIR=$HU_BASE/triton-cache
export HU_BASE HU_DATA=$HU_BASE/fineweb-edu-llama3
