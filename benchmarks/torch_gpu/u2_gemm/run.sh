#!/usr/bin/env bash
# bench.py on one leased GPU of an sm_90 node, under `research run` with cwd the source tree's root. Outside the lease:
# u2_gemm.cubin built with nvcc and held to sass_gate.py, and torch resolved by uv. Under one lease of at most MAX_MIN
# (default 15) minutes: nvidia-smi sampled every 100 ms (smi.csv) beside bench.py, whose arguments are this script's.
#   research run --on NODE --project verity --source . --cwd source --timeout 60m --env GPU_LEASE_WHO=LANE -- \
#     bash benchmarks/torch_gpu/u2_gemm/run.sh --arms lt,ci2_m2,ci2_m2+h1 --secs 4
# CUDA names the toolkit (default /usr/local/cuda-13.0).
set -uo pipefail
OUT=${OUT:?}
D=benchmarks/torch_gpu/u2_gemm
CUDA=${CUDA:-/usr/local/cuda-13.0}
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -cubin)
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/u2_gemm.cubin" "$D/u2_gemm.cu" 2> "$OUT/ptxas.txt" \
  || { cat "$OUT/ptxas.txt" >&2; exit 1; }
"$CUDA/bin/cuobjdump" -sass "$OUT/u2_gemm.cubin" > "$OUT/u2_gemm.sass"
python3 "$D/sass_gate.py" "$OUT/ptxas.txt" "$OUT/u2_gemm.sass" "$OUT/sass.json" || exit 1
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o u2_gemm.cubin u2_gemm.cu"; } > "$OUT/toolchain.txt"
UV=(uv run -q --no-project --with "torch==2.9.*" --with numpy --index-url https://download.pytorch.org/whl/cu129
    --extra-index-url https://pypi.org/simple)
CUDA_VISIBLE_DEVICES= "${UV[@]}" python -c "import torch; print(torch.__version__, torch.version.cuda)" > "$OUT/python.txt" 2>&1 \
  || { cat "$OUT/python.txt" >&2; exit 1; }
export U2_GEMM_CUBIN=$OUT/u2_gemm.cubin
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAX_MIN:-15}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit,enforced.power.limit --format=csv > "$OUT/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,power.draw.instant,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader -lms 100 > "$OUT/smi.csv" 2>&1 &
  S=$!
  "$@" --out "$OUT"
  rc=$?
  kill $S
  exit $rc' _ "${UV[@]}" python "$D/bench.py" "$@" 2>&1 | tee "$OUT/bench.log"
rc=${PIPESTATUS[0]}
echo "bench rc $rc" >> "$OUT/rc.txt"
exit "$rc"
