#!/usr/bin/env bash
# The int8 epilogue with and without a per-row subtraction (epi_fold.cu), its SASS on sm_90a counted an output
# (epi_fold_count.py): whether the fold costs no added instruction an output (premise epilogue-fold). Compile only,
# no GPU; from a pushed commit:
#
#   research run --on auto --cpus 2 --mem-gb 4 --max-min 15 --cwd source -- bash benchmarks/pouw/theory/epi_fold.sh
#
# Without nvcc on PATH it installs CUDA 13.4's compiler and cuobjdump wheels into a scratch environment.
# Exit: 0 every check holds, 1 build failed, 3 a check failed.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$OUT/sass" "$OUT/env"
NVCC=${NVCC:-$(command -v nvcc || true)}
INC=()
if [ -z "$NVCC" ]; then
  uv venv -q "$SCRATCH/v" && uv pip install -q --python "$SCRATCH/v/bin/python" nvidia-cuda-nvcc==13.4.92 \
    nvidia-cuda-cuobjdump==13.4.92 nvidia-cuda-runtime==13.4.92 nvidia-cuda-crt==13.4.92 nvidia-nvvm==13.4.92 \
    nvidia-cuda-cccl==13.3.4.3.1 || exit 1
  T=$("$SCRATCH/v/bin/python" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')
  export PATH="$T/bin:$T/nvvm/bin:$PATH"
  NVCC="$T/bin/nvcc"
  INC=(-I"$T/include")
fi
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
"$NVCC" -arch=sm_90a -cubin -O3 "${INC[@]}" -o "$SCRATCH/epi_fold.cubin" "$HERE/epi_fold.cu" >"$OUT/build.log" 2>&1 ||
  { cat "$OUT/build.log" >&2; exit 1; }
sha256sum "$SCRATCH/epi_fold.cubin" >"$OUT/env/cubin.sha256"
"$(dirname "$NVCC")/cuobjdump" -sass "$SCRATCH/epi_fold.cubin" >"$OUT/sass/epi_fold.sass"
python3 "$HERE/epi_fold_count.py" "$OUT/sass/epi_fold.sass" --json "$OUT/epi_fold.json" | tee "$OUT/table.txt"
exit "${PIPESTATUS[0]}"
