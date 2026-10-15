#!/usr/bin/env bash
# phi.cu's lookup receipt (lut) on one GPU (cwd = the source tree): every candidate gated against twins.py (two seed
# pairs) before anything is timed, then each candidate alone (pure), then the bare INT8 loop, duo, duo2, lut and the
# fill-lop3 control fused at C = 1..32 in the s8, s8pp, s8il and s8ppst modes, the SASS opcode mix, and ncu's LSU and
# shared-memory bank-conflict counts for lut and duo2 at C = 1, 2, 8 and 32. Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone \
#     --env GPU_LEASE_WHO=... --send benchmarks/receipt_phi/lut.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/lut.sh"'
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
python3 $D/sass_mix.py "$B/phi.sass" > "$O/sass-mix.json" 2>> "$O/errors.txt"
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
  "$B/phi" pure "$ITERS" "$REPS" > "$O/pure.jsonl" 2>> "$O/errors.txt" || echo "pure rc $?" >> "$O/errors.txt"
  PHI_KINDS=none,duo,duo2,lut,fill-lop3 PHI_MODES=s8,s8pp,s8il,s8ppst \
    "$B/phi" fused 16000 "$REPS" > "$O/fused.jsonl" 2>> "$O/errors.txt" || echo "fused rc $?" >> "$O/errors.txt"
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  M=smsp__inst_executed.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__inst_executed_op_shared_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg
  for row in 14 13 0; do
    for ci in 0 1 3 5; do
      ncu --metrics $M --csv -k regex:fused_k --launch-skip 1 -c 1 "$B/phi" one fused $((6 * row + ci)) 2000 2 \
        > "$O/ncu-s8-$row-$ci.csv" 2>&1 || echo "ncu s8 $row $ci rc $?" >> "$O/errors.txt"
    done
  done'
