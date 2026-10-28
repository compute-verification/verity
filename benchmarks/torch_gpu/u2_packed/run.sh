#!/usr/bin/env bash
# bench.py on one GPU of an sm_90 node whose GPUs gpu-lease hands out, with cwd the source tree's root and OUT the
# directory to write. Outside the lease: packed.cubin built with nvcc and held to sass_gate.py, and torch resolved by uv.
# Under one lease of at most MAX_MIN (default 15) minutes: nvidia-smi sampled every 100 ms (smi.csv) beside bench.py,
# whose arguments are this script's. BUILD_ONLY=1 stops after the gate.
#   OUT=DIR bash benchmarks/torch_gpu/u2_packed/run.sh --arms lt,i8,p2,p2+w4 --secs 4
# CUDA names the toolkit (default /usr/local/cuda-13.0).
set -uo pipefail
OUT=${OUT:?}
D=benchmarks/torch_gpu/u2_packed
CUDA=${CUDA:-/usr/local/cuda-13.0}
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -cubin)
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/packed.cubin" "$D/packed.cu" 2> "$OUT/ptxas.txt" \
  || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_90a -ptx -o "$OUT/packed.ptx" "$D/packed.cu"  # ptxas's line numbers
"$CUDA/bin/cuobjdump" -sass "$OUT/packed.cubin" > "$OUT/packed.sass"
python3 "$D/sass_gate.py" "$OUT/ptxas.txt" "$OUT/packed.sass" "$OUT/sass.json" || exit 1
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o packed.cubin packed.cu"; } > "$OUT/toolchain.txt"
[ "${BUILD_ONLY:-0}" = 1 ] && exit 0
UV=(uv run -q --no-project --with "torch==2.9.*" --with numpy --index-url https://download.pytorch.org/whl/cu129
    --extra-index-url https://pypi.org/simple)
CUDA_VISIBLE_DEVICES= "${UV[@]}" python -c "import torch; print(torch.__version__, torch.version.cuda)" > "$OUT/python.txt" 2>&1 \
  || { cat "$OUT/python.txt" >&2; exit 1; }
export U2_PACKED_CUBIN=$OUT/packed.cubin
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAX_MIN:-15}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm --format=csv > "$OUT/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader -lms 100 > "$OUT/smi.csv" 2>&1 &
  S=$!
  "$@" --out "$OUT"
  rc=$?
  kill $S
  exit $rc' _ "${UV[@]}" python "$D/bench.py" "$@" 2>&1 | tee "$OUT/bench.log"
rc=${PIPESTATUS[0]}
echo "bench rc $rc" >> "$OUT/rc.txt"
exit "$rc"
