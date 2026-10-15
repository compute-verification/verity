#!/usr/bin/env bash
# job_base.sh's part inside `gpu-lease 1` (env: HB, NCU, TPY, OUT, GPU_LEASE_UUID, CUDA_VISIBLE_DEVICES from the lease).
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID PEARLC_EXPECT_UUID="$GPU_LEASE_UUID"
echo "lease $GPU_LEASE_UUID index $CUDA_VISIBLE_DEVICES $(date -u +%FT%TZ)" >&2
ETA_SHAPES='32 64 128 256 512 1024 2048 8192' "$TPY" benchmarks/pouw_eta/lt_probe.py > "$OUT/lt_time.jsonl" 2> "$OUT/lt_time.err" \
  || echo "lt time rc=$?" >&2
echo "lt time done $(date -u +%FT%TZ)" >&2
KM=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__cycles_active.avg,\
sm__pipe_tensor_subpipe_hmma_cycles_active.sum,sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor_subpipe_hmma.sum,\
sm__ops_path_tensor_src_fp8_dst_fp32.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,\
smsp__issue_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active
ETA_MODE=profile ETA_SHAPES='32 128 8192' timeout 600 "$NCU" --profile-from-start off --clock-control none --cache-control none \
  --metrics "$KM" --section LaunchStats -f -o "$OUT/ncu_lt" "$TPY" benchmarks/pouw_eta/lt_probe.py > "$OUT/lt_prof.log" 2>&1 \
  || echo "ncu lt rc=$?" >&2
[ -f "$OUT/ncu_lt.ncu-rep" ] && "$NCU" --import "$OUT/ncu_lt.ncu-rep" --csv --page raw > "$OUT/ncu_lt.csv"
echo "ncu lt done $(date -u +%FT%TZ)" >&2
(
  . benchmarks/pouw/harness/jobs/env.sh "$HB" && cd benchmarks/pouw/harness && timeout 900 "$PY" bench.py --gpu lease \
    --expect-uuid "$GPU_LEASE_UUID" --clock-label locked-2100 --shapes headline --families fp8-e4m3 --out "$OUT/bench.json"
) > "$OUT/bench.log" 2>&1 || echo "bench rc=$?" >&2
echo "bench done $(date -u +%FT%TZ)" >&2
exit 0
