#!/usr/bin/env bash
# served_zk's row-commitment microbenchmark on node 2 (rows_bench.py), untimed: GPU 1's pearl_c_sm120.cubin (build.sh's nvcc
# line) and hm96_rows.cubin built on the node's CPUs, then rows_bench.py in a one-GPU preemptible lease.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/served_zk/bench.sh
# env: CALLS, REPS, PREFILL (rows_bench.py's), MAX_MIN (the lease's minutes, default 20)
set -uo pipefail
OUT=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$OUT"
WORK=/workspace/pouw/mvp-e2e; PY=$WORK/venv312/bin/python; CUDA=/usr/local/cuda-13.0
if [ "${1:-}" != gpu ]; then
  S=$(mktemp -d /workspace/pouw/tmp-served-zk-bench.XXXXXX); trap 'rm -rf "$S"' EXIT
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -o "$S/pearl_c_sm120.cubin" \
    benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu 2> "$OUT/ptxas-pearl.txt" || { tail -20 "$OUT/ptxas-pearl.txt"; exit 11; }
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -cubin -o "$S/hm96_rows.cubin" \
    benchmarks/pouw/served_zk/hm96_rows.cu 2> "$OUT/ptxas-rows.txt" || { tail -30 "$OUT/ptxas-rows.txt"; exit 12; }
  grep -A4 -E "Compiling entry function 'sz_" "$OUT/ptxas-rows.txt" | grep -E "Used|spill" | tee "$OUT/ptxas-rows-summary.txt"
  (cd "$S" && sha256sum ./*.cubin) | tee "$OUT/cubins.sha256"
  S=$S gpu-lease 1 --wait --preemptible --max-min "${MAX_MIN:-20}" -- bash "$0" gpu
  rc=$?; echo "lease exited $rc"; exit "$rc"
fi
UUID=${GPU_LEASE_UUID:?run under gpu-lease}; UUID=${UUID%%,*}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" | tee "$OUT/identity.csv"
PEARLC_EXPECT_UUID=$UUID PYTHONPATH=$PWD "$PY" benchmarks/pouw/served_zk/rows_bench.py --cubin "$S/pearl_c_sm120.cubin" \
  --rows-cubin "$S/hm96_rows.cubin" --out "$OUT/rows-bench.json" 2>&1 | tee "$OUT/rows-bench.log"
exit "${PIPESTATUS[0]}"
