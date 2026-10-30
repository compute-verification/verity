# Source me: a CPU torch venv under $HP_BASE (default ~/hardness-public), built once; single-threaded BLAS for the workers.
HP_BASE=${HP_BASE:-$HOME/hardness-public}
if [ ! -x "$HP_BASE/venv/bin/python" ]; then
  mkdir -p "$HP_BASE"
  uv venv -q --python 3.12 "$HP_BASE/venv"
  uv pip install -q --python "$HP_BASE/venv/bin/python" torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python "$HP_BASE/venv/bin/python" numpy scipy
fi
export PATH=$HP_BASE/venv/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export HP_BASE
