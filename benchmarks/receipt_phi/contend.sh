#!/usr/bin/env bash
# contend.cu on one GPU (cwd = the source tree): the exact SASS counts of every kernel's timed loops, then the sweep
# (FP8 and INT8 × 16 filler classes × F = 0..256 per 16 MMAs × W = 1, 2, 4, 8 warps per scheduler) twice on one SM and
# once on every SM, the split sweep (fillers in their own warps) on one SM, and ncu's issue, pipe and warp-state counts
# for the F = 0 loops and every class at F = 16 and 64. Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone \
#     --env GPU_LEASE_WHO=... --send benchmarks/receipt_phi/contend.sh \
#     -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/contend.sh"'
set -uxo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/receipt_phi
export B; B=$(mktemp -d /tmp/contend.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/contend" $D/contend.cu \
  > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
cuobjdump -sass "$B/contend" > "$B/contend.sass" 2>&1
(cd $D && python3 contend_sass.py "$B/contend.sass") > "$O/contend-sass.json" 2>> "$O/errors.txt"
gzip -c "$B/contend.sass" > "$O/contend.sass.gz"
export ITERS=${ITERS:-16384} REPS=${REPS:-5}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for pass in one-a one-b all; do
    g=1; [ $pass = all ] && g=0
    "$B/contend" sweep "$ITERS" "$REPS" $g > "$O/sweep-$pass.jsonl" 2>> "$O/errors.txt" \
      || echo "sweep $pass rc $?" >> "$O/errors.txt"
  done
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  M0=smsp__inst_executed.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_uniform.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg
  M=$M0,smsp__inst_issued.sum,sm__inst_executed_pipe_lsu.sum
  prof() {  # OUTFILE ARGS...: the extended metrics with the warp-state sections, else the metrics phi.sh uses
    local f=$1; shift
    ncu --metrics $M --section WarpStateStats --section SchedulerStats --csv --page raw -k regex:contend_k \
      --launch-skip 1 -c 1 "$B/contend" one "$@" > "$f" 2>&1 \
      || ncu --metrics $M0 --csv --page raw -k regex:contend_k --launch-skip 1 -c 1 "$B/contend" one "$@" > "$f" 2>&1 \
      || echo "ncu $* rc $?" >> "$O/errors.txt"
  }
  for t in 0 1; do
    for w in 1 2 4 8; do prof "$O/ncu-$t-none-0-$w.csv" $t -1 0 $w $((2048 / w)) 1; done
    for c in $(seq 0 9); do
      for f in 16 64; do prof "$O/ncu-$t-$c-$f-2.csv" $t $c $f 2 1024 1; done
    done
  done'
