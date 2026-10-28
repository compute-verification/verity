#!/usr/bin/env bash
# probe.py on one GPU of an sm_90 node whose GPUs gpu-lease hands out, with cwd the source tree's root and OUT the
# directory to write. Outside the lease: b1.cubin built with nvcc and held to sass_gate.py, and torch resolved by uv.
# Under one lease of at most MAX_MIN (default 20) minutes: nvidia-smi sampled every 100 ms (smi.csv) beside probe.py,
# whose arguments are this script's.
#   OUT=DIR bash benchmarks/torch_gpu/b1_gemm/run.sh --arms res,res+w4,res+h2 --secs 4
# CUDA names the toolkit (default /usr/local/cuda-13.0).
set -uo pipefail
OUT=${OUT:?}
D=benchmarks/torch_gpu/b1_gemm
CUDA=${CUDA:-/usr/local/cuda-13.0}
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -cubin)
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/b1.cubin" "$D/b1.cu" 2> "$OUT/ptxas.txt" \
  || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$OUT/b1.cubin" > "$OUT/b1.sass"
python3 "$D/sass_gate.py" "$OUT/ptxas.txt" "$OUT/b1.sass" "$OUT/sass.json" || exit 1
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o b1.cubin b1.cu"; } > "$OUT/toolchain.txt"
UV=(uv run -q --no-project --with "torch==2.9.*" --with numpy --index-url https://download.pytorch.org/whl/cu129
    --extra-index-url https://pypi.org/simple)
CUDA_VISIBLE_DEVICES= "${UV[@]}" python -c "import torch; print(torch.__version__, torch.version.cuda)" > "$OUT/python.txt" 2>&1 \
  || { cat "$OUT/python.txt" >&2; exit 1; }
export B1_CUBIN=$OUT/b1.cubin
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAX_MIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm --format=csv > "$OUT/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader -lms 100 > "$OUT/smi.csv" 2>&1 &
  S=$!
  "$@" --out "$OUT"
  rc=$?
  kill $S
  exit $rc' _ "${UV[@]}" python "$D/probe.py" "$@" 2>&1 | tee "$OUT/probe.log"
rc=${PIPESTATUS[0]}
echo "probe rc $rc" >> "$OUT/rc.txt"
exit "$rc"
