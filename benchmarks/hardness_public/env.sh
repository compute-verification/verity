# Source me: a CPU torch venv under $HP_BASE (default ~/hardness-public), built once; single-threaded BLAS for the workers.
HP_BASE=${HP_BASE:-$HOME/hardness-public}
mkdir -p "$HP_BASE"
( flock 9
  if [ ! -x "$HP_BASE/venv/bin/python" ]; then
    uv venv -q --python 3.12 "$HP_BASE/venv"
    uv pip install -q --python "$HP_BASE/venv/bin/python" torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    uv pip install -q --python "$HP_BASE/venv/bin/python" numpy scipy
  fi
  "$HP_BASE/venv/bin/python" -c 'import safetensors' 2>/dev/null || uv pip install -q --python "$HP_BASE/venv/bin/python" safetensors
) 9>"$HP_BASE/venv.lock"
export PATH=$HP_BASE/venv/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export HP_BASE
