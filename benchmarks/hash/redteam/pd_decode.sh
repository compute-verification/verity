#!/usr/bin/env bash
# kernels-redteam's re-run of kernels-pouw-decode at the claim's own commit, unedited (the job's cwd is its tree), with
# pd_compose.py sent as an input:
#   off the lease  Pearl-C's cubin and pd.cubin with the claim's run.sh flags; the claim's host_test.py
#   one GPU        the claim's bench.py at m16 = 32 (its check, then its interleaved timing); pd_compose.py (the check
#                  sabotaged, then a decode step's 128 linears as one graph per schedule, timed interleaved)
# Launch:
#   research run --on vy-nebius-2 --project verity --campaign hash-loop --source CLAIM_TREE --cwd source \
#     --send benchmarks/hash/redteam/pd_compose.py --send benchmarks/hash/redteam/pd_decode.sh \
#     --env GPU_LEASE_WHO=kernels-redteam -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/pd_decode.sh"'
set -euo pipefail
: "${OUT:?run under research run, which sets OUT}"
D=$PWD/benchmarks/hash/pouw_decode
CUDA=${CUDA:-/usr/local/cuda}
SCR=$(mktemp -d); trap 'rm -rf "$SCR"' EXIT
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin)
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]}"; } > "$OUT/toolchain.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$SCR/pearl_c_sm120.cubin" benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu \
  2> "$OUT/ptxas_pearl_c_sm120.txt" & p1=$!
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$SCR/pd.cubin" "$D/pd.cu" 2> "$OUT/ptxas_pd.txt" & p2=$!
( g++ -O2 -std=c++20 -pthread -shared -fPIC -Wno-unknown-pragmas -o "$SCR/pd_host.so" "$D/pd_host.cpp" &&
  PYTHONPATH=$PWD python3 "$D/host_test.py" "$SCR/pd_host.so" > "$OUT/host_test.txt" 2>&1 ) & p3=$!
wait "$p1"; wait "$p2"
wait "$p3" || echo "host_test: exit $?" >> "$OUT/host_test.txt"
tail -2 "$OUT/host_test.txt"
(cd "$SCR" && sha256sum ./*.cubin) > "$OUT/cubins.sha256"
export PD_SIDE=851,979 SCR D
gpu-lease 1 --wait --preemptible --max-min 20 -- bash -c '
  set -euo pipefail
  q=index,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw
  nvidia-smi --query-gpu=$q --format=csv > "$OUT/gpu.csv"
  python3 "$D/bench.py" "$SCR/pearl_c_sm120.cubin" "$SCR/pd.cubin" --out "$OUT" --reps 30 --rows 32 --shapes qkv,o,gate_up,down
  python3 "$RESEARCH_RUN_DIR/inputs/pd_compose.py" "$D" "$SCR/pearl_c_sm120.cubin" "$SCR/pd.cubin" --out "$OUT" --layers 32 --reps 30
  nvidia-smi --query-gpu=$q --format=csv > "$OUT/gpu_end.csv"
'
python3 "$D/report.py" "$OUT" > "$OUT/report.txt" 2>&1 || true
cat "$OUT/report.txt"
