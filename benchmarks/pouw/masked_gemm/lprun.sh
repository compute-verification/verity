#!/usr/bin/env bash
# One-side-peeled noise (lpgemm.cu's header) on one leased sm_120 GPU (cwd = the clone's root): the build outside any
# lease; then one short lease for the device gates (five small shapes) and the interleaved timing (the Llama linears at
# m = 8192, 4096^3, Q K^T and P V at 32 heads, d = 128, s = 2048 and 8192); after it, lpref.py checks every gated arm
# against hashlib and numpy, and lptable.py renders the table (an arm whose gate failed is marked). Clocks are read,
# never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd clone --env GPU_LEASE_WHO=LANE \
#     --send benchmarks/pouw/masked_gemm/lprun.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/lprun.sh"'
# STAGE=build|all (default all); REPS (default 7); SHAPES; LP_ARMS.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/masked_gemm
export B; B=$(mktemp -d /tmp/lp-noise.XXXXXX); trap 'rm -rf "$B"' EXIT
STAGE=${STAGE:-all}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/lpgemm" $D/lpgemm.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build-lpgemm.log" 2>&1
rc=$?
echo "build lpgemm rc $rc" | tee -a "$O/rc.txt"
grep -E "error|warning" "$O/build-lpgemm.log" | head -40
grep -E "spill|Used" "$O/build-lpgemm.log" > "$O/regs-lpgemm.txt"
grep -B2 -E "[1-9][0-9]* bytes spill" "$O/build-lpgemm.log" | grep -E "entry|spill" | head -20
[ "$rc" = 0 ] || { tail -40 "$O/build-lpgemm.log"; exit 4; }
[ "$STAGE" = build ] && exit 0

export REPS=${REPS:-7}
export SHAPES=${SHAPES:-"qkv:8192:6144:4096:1 o:8192:4096:4096:1 gate_up:8192:28672:4096:1 down:8192:4096:14336:1 sq4096:4096:4096:4096:1 qk2k:2048:2048:128:32 qk8k:8192:8192:128:32 pv2k:2048:128:2048:32 pv8k:8192:128:8192:32"}
mkdir -p "$B"/g{1,2,3,4,5}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 12 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  { "$B/lpgemm" gate "$B/g1" 256 256 512 1 0x5EED1234 3 && "$B/lpgemm" gate "$B/g2" 256 128 128 2 7 2 &&
    "$B/lpgemm" gate "$B/g3" 128 256 384 3 99 4 && "$B/lpgemm" gate "$B/g4" 128 128 14336 1 5 1 &&
    "$B/lpgemm" gate "$B/g5" 256 128 2048 2 11 3; } > "$O/gate-lpgemm.txt" 2>&1
  echo "gate lpgemm rc $?" >> "$O/rc.txt"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  # shellcheck disable=SC2086
  "$B/lpgemm" bench "$REPS" $SHAPES > "$O/bench.jsonl" 2> "$O/bench.err"
  echo "bench lpgemm rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/gate-lpgemm.txt"
for g in g1 g2 g3 g5; do python3 $D/lpref.py "$B/$g" >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "lpref $g rc $?" >> "$O/rc.txt"; done
python3 $D/lpref.py "$B/g4" --no-receipts >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "lpref g4 rc $?" >> "$O/rc.txt"
python3 -c 'import json,sys
for l in open(sys.argv[1]): j = json.loads(l); print(j["dir"], "ok" if j["ok"] else "FAILED", j["n_checks"], "checks; failed:", j["failed"][:40], "unchecked:", j["unchecked_arms"])' \
  "$O/gate.jsonl"
tail -20 "$O/gate.err"
python3 $D/lptable.py "$O/bench.jsonl" "$O/gate.jsonl" > "$O/table.md" 2> "$O/table.err"
cat "$O/rc.txt" "$O/table.md" "$O/table.err"
