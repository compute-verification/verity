# Source me: a CPU torch venv and the TinyStories token cache under $HML_BASE (default ~/hardness-ml), built once.
HML_BASE=${HML_BASE:-$HOME/hardness-ml}
HML_DATA=$HML_BASE/tinystories-bpe4096
if [ ! -x "$HML_BASE/venv/bin/python" ]; then
  mkdir -p "$HML_BASE"
  uv venv -q --python 3.12 "$HML_BASE/venv"
  uv pip install -q --python "$HML_BASE/venv/bin/python" torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python "$HML_BASE/venv/bin/python" numpy tokenizers huggingface_hub
fi
export PATH=$HML_BASE/venv/bin:$PATH
export HML_BASE HML_DATA
