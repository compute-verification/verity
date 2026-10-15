#!/usr/bin/env bash
# filler.cu on one GPU (cwd = the source tree): the build and each loop's SASS opcode mix; the gate (every kernel's
# receipts and accumulators against twin.py, two seed and key pairs) before anything is timed; the sweep (the filler
# alone by occupancy, method b's in-loop extras and method a's co-resident probes, each beside a filler-alone launch);
# ncu on the filler alone and a few probe and extra configurations, NCU_REPS times (method c, clocks left alone:
# --clock-control none); then, after the lease, the timed launches' sampled receipts against twin.py. Clocks and power
# are sampled every 200 ms throughout the lease. Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-12 --source . --cwd source \
#     --env GPU_LEASE_WHO=W11 -- bash benchmarks/filler_slack/run.sh
set -uxo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/filler_slack
export B; B=$(mktemp -d /tmp/filler-slack.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/filler" $D/filler.cu > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1 || true
cuobjdump -sass "$B/filler" > "$B/filler.sass" 2>&1
python3 $D/sass_loops.py "$B/filler.sass" > "$O/sass-loops.json"
gzip -c "$B/filler.sass" > "$O/filler.sass.gz"
export ITERS=${ITERS:-32000} REPS=${REPS:-7} NCU_REPS=${NCU_REPS:-5} NCU_ITERS=${NCU_ITERS:-8000}
mkdir -p "$O/ncu"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  q() {
    nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,power.limit,temperature.gpu \
      --format=csv,noheader -i "$GPU_LEASE_UUID" 2>&1
    nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader -i "$GPU_LEASE_UUID" 2>&1
  }
  echo "start $(date -u +%FT%TZ)" >> "$O/gpu.txt"; q >> "$O/gpu.txt"
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader \
    -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  ok=1
  for sk in "0x5EED1234 0xC0FFEE11" "7 99"; do
    tag=$(echo "$sk" | tr " " "-")
    "$B/filler" gate $sk > "$B/gate-$tag.txt" 2>> "$O/errors.txt" || ok=0
    python3 $D/gate.py < "$B/gate-$tag.txt" >> "$O/gate.jsonl" 2>> "$O/errors.txt" || ok=0
    gzip -c "$B/gate-$tag.txt" > "$O/gate-$tag.txt.gz"
  done
  if [ $ok != 1 ]; then echo "gate failed: not timed" >> "$O/errors.txt"; kill $smi; exit 6; fi
  echo "gate-ok $(date -u +%FT%TZ)" >> "$O/gpu.txt"; q >> "$O/gpu.txt"
  "$B/filler" sweep "$ITERS" "$REPS" "$O/verify.txt" > "$O/sweep.jsonl" 2>> "$O/errors.txt" || echo "sweep rc $?" >> "$O/errors.txt"
  echo "sweep-done $(date -u +%FT%TZ)" >> "$O/gpu.txt"; q >> "$O/gpu.txt"
  M=smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__cycles_active.sum,smsp__issue_active.avg.pct_of_peak_sustained_active
  M=$M,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum
  M=$M,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_uniform.sum,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active
  M=$M,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active
  M=$M,sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
  M=$M,sm__cycles_elapsed.avg
  for r in $(seq 1 "$NCU_REPS"); do
    for cfg in "base-u1 16 0 0" "base-u1 8 0 0" "base-u4 16 0 0" "x-iadd3-u1-x4 16 0 0" "x-imad-u1-x4 16 0 0" \
               "p-iadd3 16 4 0" "p-imad 16 4 0" "p-iadd3 16 4 1600"; do
      set -- $cfg
      ncu --clock-control none --metrics "$M" --csv -k regex:fk -c 1 "$B/filler" one "$1" "$2" "$3" "$4" "$NCU_ITERS" \
        > "$O/ncu/m-$1-fw$2-pw$3-s$4-r$r.csv" 2>&1 || echo "ncu $cfg r$r rc $?" >> "$O/errors.txt"
    done
  done
  for cfg in "base-u1 16 0 0" "base-u1 8 0 0"; do
    set -- $cfg
    ncu --clock-control none --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis \
      --section InstructionStats --csv --page details -k regex:fk -c 1 "$B/filler" one "$1" "$2" "$3" "$4" "$NCU_ITERS" \
      > "$O/ncu/sections-$1-fw$2.csv" 2>&1 || echo "ncu sections $cfg rc $?" >> "$O/errors.txt"
  done
  echo "end $(date -u +%FT%TZ)" >> "$O/gpu.txt"; q >> "$O/gpu.txt"
  kill $smi'
rc=$?
if [ -s "$O/verify.txt" ]; then
  python3 $D/gate.py < "$O/verify.txt" > "$O/verify.json" || echo "verify of the timed launches failed" >> "$O/errors.txt"
fi
python3 $D/analyze.py "$O" > "$O/summary.json" 2>> "$O/errors.txt" || echo "analyze rc $?" >> "$O/errors.txt"
exit $rc
