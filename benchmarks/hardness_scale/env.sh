# Source me: the CUDA torch venv and the token cache under $HSC_BASE, built once (vy-nebius-1, RTX PRO 6000, driver 580).
HSC_BASE=${HSC_BASE:-/workspace/research/lanes/compute-research/hardness-scale}
HSC_DATA=$HSC_BASE/fineweb-edu-10bt-smollm2
if [ ! -x "$HSC_BASE/venv/bin/python" ]; then
  mkdir -p "$HSC_BASE"
  uv venv -q --python 3.12 "$HSC_BASE/venv"
  uv pip install -q --python "$HSC_BASE/venv/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
  uv pip install -q --python "$HSC_BASE/venv/bin/python" numpy tokenizers huggingface_hub hf_xet pyarrow
fi
export PATH=$HSC_BASE/venv/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=$HSC_BASE/inductor-cache TRITON_CACHE_DIR=$HSC_BASE/triton-cache
export HSC_BASE HSC_DATA
