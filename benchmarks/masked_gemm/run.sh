#!/usr/bin/env bash
# The honest cost of masked matmul PoUW (mgemm.cu's header) on one leased sm_120 GPU (cwd = the clone's root): build
# outside any lease; the gate (every arm at three shapes, poisoned outputs, the third at k = 14336 for the accumulation's
# exactness at depth) under a short lease, checked by mref.py after it; then the interleaved timing at the served shapes
# under a second lease, whatever the gate said (the table marks an arm whose gate failed). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone --env GPU_LEASE_WHO=T1 \
#     --send benchmarks/masked_gemm/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
# STAGE=build|gate|all (default all); REPS (default 11); SHAPES (name:m:n:k); MG_ARMS.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/masked_gemm
export B; B=$(mktemp -d /tmp/masked-gemm.XXXXXX); trap 'rm -rf "$B"' EXIT
STAGE=${STAGE:-all}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/mgemm" $D/mgemm.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build.log" 2>&1 || { grep -E "error|warning" "$O/build.log" | head -60; tail -30 "$O/build.log"; exit 4; }
grep -A1 -E "Compiling entry function '_Z[0-9]+(mi|mf)_" "$O/build.log" | grep -E "entry|spill" | head -40
grep -E "spill|Used" "$O/build.log" > "$O/regs.txt"
[ "$STAGE" = build ] && exit 0

mkdir -p "$B/g1" "$B/g2" "$B/g3"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 10 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm --format=csv,noheader -i "$GPU_LEASE_UUID" > "$O/gpu-gate.txt"
  "$B/mgemm" gate "$B/g1" 256 256 512 0x5EED1234 && "$B/mgemm" gate "$B/g2" 384 256 1024 7 &&
  "$B/mgemm" gate "$B/g3" 128 256 14336 99' > "$O/gate-gpu.txt" 2>&1
echo "gate device rc $?" >> "$O/rc.txt"
cat "$O/gate-gpu.txt"
python3 $D/mref.py "$B/g1" >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "mref g1 rc $?" >> "$O/rc.txt"
python3 $D/mref.py "$B/g2" >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "mref g2 rc $?" >> "$O/rc.txt"
python3 $D/mref.py "$B/g3" --no-receipts >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "mref g3 rc $?" >> "$O/rc.txt"
python3 -c 'import json,sys
for l in open(sys.argv[1]): j = json.loads(l); print(j["dir"], j["m"], j["n"], j["k"], "ok" if j["ok"] else "FAILED", j["n_checks"], "checks; failed:", j["failed"][:40], "unchecked:", j["unchecked_arms"])' "$O/gate.jsonl"
cat "$O/gate.err" | tail -20
[ "$STAGE" = gate ] && exit 0

export REPS=${REPS:-11}
export SHAPES=${SHAPES:-"qkv:8192:6144:4096 o:8192:4096:4096 gate_up:8192:28672:4096 down:8192:4096:14336 sq4096:4096:4096:4096"}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 40 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  # shellcheck disable=SC2086
  "$B/mgemm" bench "$REPS" $SHAPES > "$O/bench.jsonl" 2> "$O/bench.err"
  echo "bench rc $?" >> "$O/rc.txt"
  kill $smi'
python3 $D/mtable.py "$O/bench.jsonl" "$O/gate.jsonl" > "$O/table.md" 2> "$O/table.err"
cat "$O/rc.txt" "$O/table.md" "$O/table.err"
