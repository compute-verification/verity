#!/usr/bin/env bash
# Where the e2e slice's edited GEMM spends its time at o_proj (m = 8192, n = k = 4096), on one leased sm_120 GPU (cwd = the
# source tree's root). Builds prof.cu outside the lease; under it: lpgemm's interleaved items for every part (LP_ARMS), the
# slice's per-call interleaving of the routes against plain cuBLASLt, and ncu's pipe and stall counters on the kernels.
# Clocks are read, never set (ncu --clock-control none).
#
#   research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/edited_gemm/profile.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export BIN=$(mktemp -d /tmp/egemm-prof.XXXXXX)
trap 'rm -rf "$BIN"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/prof" $D/prof.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1
rc=$?
echo "build rc $rc" | tee -a "$O/rc.txt"
[ "$rc" = 0 ] || { tail -40 "$O/build.log"; exit 4; }
export REPS=${REPS:-9}
export LP_ARMS=${LP_ARMS:-plain,cplain,cduo,genU,genF,formA,formB,speel,bsp,tK,t_bfus,t_bsep,td_bfus,td_bsep,bonly,lt_bfus}
export IL_ARMS=${IL_ARMS:-"td_bfus t_bfus lt_bfus cduo cplain tK"}
export NCU_ARMS=${NCU_ARMS:-"plain cplain cduo tK td_bfus t_bfus"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 14 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$BIN/prof" bench "$REPS" o:8192:4096:4096:1 > "$O/bench.jsonl" 2> "$O/bench.err"; echo "bench rc $?" >> "$O/rc.txt"
  for a in $IL_ARMS; do
    "$BIN/prof" il 41 $a 8192 4096 4096 >> "$O/il.jsonl" 2>> "$O/il.err"; echo "il $a rc $?" >> "$O/rc.txt"
  done
  kill $smi
  M=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,lts__t_bytes.sum,dram__bytes.sum
  M2=sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_uniform.sum,smsp__warps_eligible.avg.per_cycle_active,smsp__warps_active.avg.per_cycle_active
  for a in $NCU_ARMS; do
    ncu --clock-control none --profile-from-start off --metrics $M --csv "$BIN/prof" one $a 8192 4096 4096 \
      > "$O/ncu-$a.csv" 2>&1 || echo "ncu $a rc $?" >> "$O/rc.txt"
    ncu --clock-control none --profile-from-start off --metrics $M2 --csv "$BIN/prof" one $a 8192 4096 4096 \
      > "$O/ncu2-$a.csv" 2>&1 || echo "ncu2 $a rc $?" >> "$O/rc.txt"
    ncu --clock-control none --profile-from-start off --csv \
      --metrics "regex:^smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$" \
      "$BIN/prof" one $a 8192 4096 4096 > "$O/ncu-stall-$a.csv" 2>&1 || echo "ncu stall $a rc $?" >> "$O/rc.txt"
  done'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
python3 $D/prof_summary.py "$O" | tee "$O/summary.md"
