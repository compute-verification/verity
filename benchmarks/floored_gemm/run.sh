#!/usr/bin/env bash
# fgemm.cu on one leased sm_120 GPU (cwd = the clone's root): the build and its SASS counts outside any lease; the gate
# (every arm's outputs and receipts at three small shapes, both tile configs, poisoned outputs) dumped under a short lease
# and checked on the CPU by ref.py after it; only if every check passes, the interleaved timing at the served shapes and
# ncu's tensor-pipe counts under a second lease. Clocks are read, never set (ncu --clock-control none). Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-12 --source . --cwd clone --env GPU_LEASE_WHO=W13 \
#     --send benchmarks/floored_gemm/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
# STAGE=build|gate|all (default all); REPS (default 15); SHAPES (name:m:n:k:C ..., C the tile config P, Q, W or D); NCU_SHAPES (same).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/floored_gemm
export B; B=$(mktemp -d /tmp/floored-gemm.XXXXXX); trap 'rm -rf "$B"' EXIT
STAGE=${STAGE:-all}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/fgemm" $D/fgemm.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1 || { tail -80 "$O/build.log"; exit 4; }
grep -E "spill|Used" "$O/build.log" | head -60
cuobjdump -sass "$B/fgemm" > "$B/fgemm.sass"
python3 $D/sass_mix.py "$B/fgemm.sass" > "$O/sass-mix.json" || { echo "no IMMA in the binary"; exit 5; }
gzip -c "$B/fgemm.sass" > "$O/fgemm.sass.gz"
[ "$STAGE" = build ] && exit 0

mkdir -p "$B/g1" "$B/g2" "$B/g3" "$B/g4" "$B/g5" "$B/g6"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm --format=csv,noheader -i "$GPU_LEASE_UUID" > "$O/gpu-gate.txt"
  "$B/fgemm" gate "$B/g1" 256 256 512 0 0x5EED1234 P && "$B/fgemm" gate "$B/g2" 384 256 256 2 7 P &&
  "$B/fgemm" gate "$B/g3" 128 96 384 2 99 D && "$B/fgemm" gate "$B/g4" 256 128 1024 3 0xC0FFEE11 D &&
  "$B/fgemm" gate "$B/g5" 384 192 512 4 0xF100D Q &&
  "$B/fgemm" gate "$B/g6" 384 256 512 2 0xABCDE1 W' > "$O/gate-gpu.txt" 2>&1 \
  || { cat "$O/gate-gpu.txt"; echo "gate (device) failed: not timed"; exit 6; }
for g in g1 g2 g3 g4 g5 g6; do
  python3 $D/ref.py "$B/$g" >> "$O/gate.jsonl" 2>> "$O/gate.err" || { cat "$O/gate.jsonl" "$O/gate.err"; echo "gate $g failed: not timed"; exit 6; }
done
cat "$O/gate.jsonl"
[ "$STAGE" = gate ] && exit 0

export REPS=${REPS:-15}
# prefill shapes on 128 x 128 tiles of 8 warps (P) and of 16 (W), and on 128 x 64 (Q); decode on 128 x 32 (D)
export SHAPES=${SHAPES:-"p_qkv:8192:6144:4096:P p_o:8192:4096:4096:P p_gate_up:8192:28672:4096:P p_down:8192:4096:14336:P sq4096:4096:4096:4096:P q_qkv:8192:6144:4096:Q q_o:8192:4096:4096:Q q_gate_up:8192:28672:4096:Q q_down:8192:4096:14336:Q qsq4096:4096:4096:4096:Q w_qkv:8192:6144:4096:W w_o:8192:4096:4096:W w_gate_up:8192:28672:4096:W w_down:8192:4096:14336:W wsq4096:4096:4096:4096:W d_qkv:128:6144:4096:D d_o:128:4096:4096:D d_gate_up:128:28672:4096:D d_down:128:4096:14336:D"}
export NCU_SHAPES=${NCU_SHAPES:-"p_o:8192:4096:4096:P q_o:8192:4096:4096:Q w_o:8192:4096:4096:W sq4096:4096:4096:4096:P d_o:128:4096:4096:D"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  # shellcheck disable=SC2086
  "$B/fgemm" bench "$REPS" $SHAPES > "$O/bench.jsonl" 2> "$O/bench.err"
  echo "bench rc $?" >> "$O/rc.txt"
  kill $smi
  M=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed
  for s in $NCU_SHAPES; do
    IFS=: read -r nm m n k c <<< "$s"
    for arm in lt plain g0 c d c2; do
      ncu --clock-control none --profile-from-start off --metrics $M --csv "$B/fgemm" one $arm $m $n $k $c \
        > "$O/ncu-$nm-$arm.csv" 2>&1 || echo "ncu $nm $arm rc $?" >> "$O/rc.txt"
    done
  done
  for c in Q W; do
    for arm in g0 c c2; do
      ncu --clock-control none --profile-from-start off --section WarpStateStats --section SchedulerStats --csv \
        "$B/fgemm" one $arm 8192 4096 4096 $c > "$O/ncu-stall-$c-o-$arm.csv" 2>&1 || echo "ncu stall $c $arm rc $?" >> "$O/rc.txt"
      ncu --clock-control none --profile-from-start off --csv \
        --metrics "regex:^smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio$" \
        "$B/fgemm" one $arm 8192 4096 4096 $c > "$O/ncu-stallr-$c-o-$arm.csv" 2>&1 || echo "ncu stallr $c $arm rc $?" >> "$O/rc.txt"
    done
  done'
python3 $D/table.py "$O/bench.jsonl" "$O" > "$O/table.md" 2> "$O/table.err"
cat "$O/rc.txt" "$O/table.md"
