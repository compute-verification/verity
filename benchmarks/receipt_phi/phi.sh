#!/usr/bin/env bash
# phi.cu on one GPU (cwd = the source tree): every receipt φ candidate gated against twins.py (two seed pairs) before
# anything is timed, then the pipe probes, each candidate alone (pure), each fused into an mma.sync loop at
# C = 1, 2, 4, 8, 16, 32 MMAs per absorb (FP8 or INT8; serial, ping-pong or interleaved accumulators, INT8 also with
# staggered warps; with the controls) at the occupancy each kernel's registers allow and again at 2 blocks per SM, the
# SASS opcode mix of every timed loop, and ncu's pipe counts for the probes, the pure loops, the C = 1 fused loops and
# the C = 8 s8pp, s8il and s8ppst loops.
# Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone \
#     --env GPU_LEASE_WHO=... --send benchmarks/receipt_phi/phi.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/phi.sh"'
set -uxo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/receipt_phi
export B; B=$(mktemp -d /tmp/receipt-phi.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/phi" $D/phi.cu > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
cuobjdump -sass "$B/phi" > "$B/phi.sass" 2>&1
python3 $D/sass_mix.py "$B/phi.sass" > "$O/sass-mix.json"
gzip -c "$B/phi.sass" > "$O/phi.sass.gz"
export ITERS=${ITERS:-20000} REPS=${REPS:-5}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  ok=1
  for sk in "0x5EED1234 0xC0FFEE11" "7 99"; do
    "$B/phi" gate $sk > "$B/gate.txt" 2>> "$O/errors.txt" || ok=0
    python3 $D/gate.py < "$B/gate.txt" >> "$O/gate.jsonl" 2>> "$O/errors.txt" || ok=0
  done
  [ $ok = 1 ] || { echo "gate failed: not timed" >> "$O/errors.txt"; exit 6; }
  "$B/phi" probe "$ITERS" "$REPS" > "$O/probe.jsonl" 2>> "$O/errors.txt" || echo "probe rc $?" >> "$O/errors.txt"
  "$B/phi" pure "$ITERS" "$REPS" > "$O/pure.jsonl" 2>> "$O/errors.txt" || echo "pure rc $?" >> "$O/errors.txt"
  "$B/phi" fused 16000 "$REPS" > "$O/fused.jsonl" 2>> "$O/errors.txt" || echo "fused rc $?" >> "$O/errors.txt"
  "$B/phi" fused 16000 "$REPS" 2 > "$O/fused-occ2.jsonl" 2>> "$O/errors.txt" || echo "fused occ2 rc $?" >> "$O/errors.txt"
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  M=smsp__inst_executed.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_uniform.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg
  for p in $(seq 0 17); do
    ncu --metrics $M --csv -k regex:probe_k --launch-skip 1 -c 1 "$B/phi" one probe $p 2000 > "$O/ncu-probe-$p.csv" 2>&1 \
      || echo "ncu probe $p rc $?" >> "$O/errors.txt"
  done
  for k in $(seq 0 11); do
    ncu --metrics $M --csv -k regex:pure_k --launch-skip 1 -c 1 "$B/phi" one pure $k 2000 > "$O/ncu-pure-$k.csv" 2>&1 \
      || echo "ncu pure $k rc $?" >> "$O/errors.txt"
  done
  for m in 0 1 2 3 4 5; do
    for k in $(seq 0 16); do
      ncu --metrics $M --csv -k regex:fused_k --launch-skip 1 -c 1 "$B/phi" one fused $((6 * k)) 2000 $m \
        > "$O/ncu-fused-$m-$k.csv" 2>&1 || echo "ncu fused $m $k rc $?" >> "$O/errors.txt"
    done
  done
  for m in 3 5 7; do
    for k in $(seq 0 16); do
      ncu --metrics $M --csv -k regex:fused_k --launch-skip 1 -c 1 "$B/phi" one fused $((6 * k + 3)) 2000 $m 2 \
        > "$O/ncu-fused-c8-$m-$k.csv" 2>&1 || echo "ncu fused c8 $m $k rc $?" >> "$O/errors.txt"
    done
  done'
