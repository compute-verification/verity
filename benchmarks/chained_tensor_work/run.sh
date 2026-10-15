#!/usr/bin/env bash
# chain.cu on one leased sm_120 GPU (cwd = the clone's root). Outside any lease: the build, the SASS (the plain kernels'
# epilogue check, sass_epilogue.py) and gen.py's keys. Under a short lease: the gate dumps at three small shapes, checked on
# the CPU by ref.py after it; only if every check passes, the interleaved timing and ncu's tensor-pipe counts under a
# second lease. Then the verifier's cost on the CPU (verify.py). Clocks are read, never set (ncu --clock-control none).
# Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-12 --source . --cwd clone --env GPU_LEASE_WHO=T2 \
#     --send benchmarks/chained_tensor_work/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
# STAGE=build|gate|all (default all); REPS (default 25); SHAPES (name:m:k:S ...); ARMS (chain.cu's); NCU (default 1);
# VERIFY (M:K ..., or none)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/chained_tensor_work
export PYTHONPATH=$PWD:$PWD/$D
export B; B=$(mktemp -d /tmp/chained-tensor-work.XXXXXX); trap 'rm -rf "$B"' EXIT
STAGE=${STAGE:-all}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/chain" $D/chain.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1 || { tail -80 "$O/build.log"; exit 4; }
grep -E "spill|Used" "$O/build.log" | grep -B1 -A1 plain | head -20
cuobjdump -sass "$B/chain" > "$B/chain.sass"
python3 $D/sass_epilogue.py "$B/chain.sass" > "$O/sass-epilogue.json"; echo "sass_epilogue rc $?" | tee -a "$O/rc.txt"
cat "$O/sass-epilogue.json"
grep -A400 "Function : .*p_plain" "$B/chain.sass" | gzip -c > "$O/p_plain.sass.gz"
grep -A400 "Function : .*w_plain" "$B/chain.sass" | gzip -c > "$O/w_plain.sass.gz"
export SHAPES=${SHAPES:-"big:8192:8192:32 mid:8192:4096:48 small:4096:4096:64"}
python3 $D/gen.py "$B/keys" 64 128 256 512 4096 8192 || { echo "gen failed"; exit 4; }
[ "$STAGE" = build ] && exit 0

mkdir -p "$B/g1" "$B/g2" "$B/g3"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm --format=csv,noheader -i "$GPU_LEASE_UUID" > "$O/gpu-gate.txt"
  "$B/chain" gate "$B/g1" "$B/keys" 256 1024 4 0x5EED1234 && "$B/chain" gate "$B/g2" "$B/keys" 128 2048 3 7 &&
  "$B/chain" gate "$B/g3" "$B/keys" 512 4096 2 0xC0FFEE11' > "$O/gate-gpu.txt" 2>&1 \
  || { cat "$O/gate-gpu.txt"; echo "gate (device) failed: not timed"; exit 6; }
for g in g1 g2 g3; do
  python3 $D/ref.py "$B/$g" >> "$O/gate.jsonl" 2>> "$O/gate.err" || { cat "$O/gate.jsonl" "$O/gate.err"; echo "gate $g failed: not timed"; exit 6; }
done
cat "$O/gate.jsonl"
[ "$STAGE" = gate ] && exit 0

export REPS=${REPS:-25} NCU=${NCU:-1}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 30 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  # shellcheck disable=SC2086
  "$B/chain" bench "$B/keys" "$REPS" $SHAPES > "$O/bench.jsonl" 2> "$O/bench.err"
  echo "bench rc $?" >> "$O/rc.txt"
  kill $smi
  M=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed
  [ "$NCU" = 1 ] && for arm in lt/0 kP/0 kW/0 hash; do
    a=${arm//\//_}
    ncu --clock-control none --profile-from-start off --launch-count 12 --metrics $M --csv "$B/chain" one "$B/keys" $arm 8192 8192 4 \
      > "$O/ncu-big-$a.csv" 2>&1 || echo "ncu big $arm rc $?" >> "$O/rc.txt"
  done' > "$O/lease-bench.txt" 2>&1
cat "$O/rc.txt"
head -c 3000 "$O/bench.jsonl" | cut -c1-400

for v in ${VERIFY:-"8192:8192 4096:4096"}; do
  [ "$v" = none ] && continue
  IFS=: read -r vm vk <<< "$v"
  python3 $D/verify.py "$vm" "$vk" 1 8 64 128 >> "$O/verify.jsonl" 2>> "$O/verify.err" || echo "verify $v rc $?" >> "$O/rc.txt"
done
cat "$O/verify.jsonl"
