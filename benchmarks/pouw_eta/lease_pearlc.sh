#!/usr/bin/env bash
# job_pearlc.sh's part inside `gpu-lease 1` (env: SHIP, NCU, OUT, GPU_LEASE_UUID, CUDA_VISIBLE_DEVICES from the lease).
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID PEARLC_EXPECT_UUID="$GPU_LEASE_UUID"
echo "lease $GPU_LEASE_UUID index $CUDA_VISIBLE_DEVICES $(date -u +%FT%TZ)" >&2
cd "$SHIP"
VARIANTS=sm120 SHAPES='32x8192x8192 128x8192x8192 8192x8192x8192' REPS=9 python3 run.py check/check.json \
  > "$OUT/run_py.jsonl" 2> "$OUT/run_py.err"
rc=$?
tail -5 "$OUT/run_py.err" >&2
if [ "$rc" -ne 0 ]; then echo "run.py rc=$rc: the gate failed, nothing timed" >&2; exit "$rc"; fi
echo "gate passed $(date -u +%FT%TZ)" >&2
ETA_SHAPES='32x8192x8192 64x8192x8192 128x8192x8192 128x8192x8192@64 256x8192x8192 512x8192x8192 1024x8192x8192 2048x8192x8192 8192x8192x8192' \
  python3 eta.py > "$OUT/eta_time.jsonl" 2> "$OUT/eta_time.err" || echo "eta time rc=$?" >&2
echo "eta time done $(date -u +%FT%TZ)" >&2
ETA_MODE=clock python3 eta.py > "$OUT/eta_clock.jsonl" 2> "$OUT/eta_clock.err" || echo "eta clock rc=$?" >&2
echo "clock done $(date -u +%FT%TZ)" >&2
KM=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__cycles_active.avg,\
sm__pipe_tensor_subpipe_hmma_cycles_active.sum,sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor_subpipe_hmma.sum,\
sm__ops_path_tensor_src_fp8_dst_fp32.sum,sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,\
smsp__issue_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__inst_executed.sum
GM=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__pipe_tensor_subpipe_hmma_cycles_active.sum,\
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__ops_path_tensor_src_fp8_dst_fp32.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_elapsed
ETA_MODE=profile ETA_W=1 timeout 600 "$NCU" --profile-from-start off --clock-control none --cache-control none --metrics "$KM" \
  --section LaunchStats -f -o "$OUT/ncu_kernels" python3 eta.py > "$OUT/eta_prof_kernels.log" 2>&1 || echo "ncu kernels rc=$?" >&2
echo "ncu kernels done $(date -u +%FT%TZ)" >&2
ETA_MODE=profile ETA_GRAPHS=1 ETA_W=3 timeout 600 "$NCU" --profile-from-start off --clock-control none --cache-control none \
  --graph-profiling graph --metrics "$GM" -f -o "$OUT/ncu_graphs" python3 eta.py > "$OUT/eta_prof_graphs.log" 2>&1 \
  || echo "ncu graphs rc=$?" >&2
echo "ncu graphs done $(date -u +%FT%TZ)" >&2
exit 0
