#!/usr/bin/env bash
# Cutting P V's noise cost (pvgemm.cu's header) and u17 from 32-bit words (sgemm.cu's u17w; u17p and u17wp the same
# noise five lanes a patch) on one leased sm_120 GPU (cwd = the clone's root): both builds outside any lease; then one
# short lease for the device gates (sgemm at four small shapes, pvgemm at three) and the interleaved timings (sgemm's
# alphabets at the edited-GEMM table's shapes, pvgemm at s = 2048 and 8192, H = 32); after it, sref.py and pvref.py
# check every gated arm against hashlib and numpy, and stable.py and pvtable.py render the tables (an arm whose gate
# failed is marked). Clocks are read, never set.
#
#   research run --on MACHINE --project verity --source . --cwd clone --env GPU_LEASE_WHO=LANE \
#     --send benchmarks/pouw/masked_gemm/pvrun.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/pvrun.sh"'
# STAGE=build|all (default all); REPS (default 7); PV_S (default "2048 8192"); SHAPES; SG_ARMS; PV_ARMS.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/masked_gemm
export B; B=$(mktemp -d /tmp/pv-noise.XXXXXX); trap 'rm -rf "$B"' EXIT
STAGE=${STAGE:-all}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
NV=(nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v)
"${NV[@]}" -o "$B/sgemm" $D/sgemm.cu -lcuda -lcublasLt -lnvidia-ml > "$O/build-sgemm.log" 2>&1 &
sp=$!
"${NV[@]}" -o "$B/pvgemm" $D/pvgemm.cu -lcuda -lcublasLt -lnvidia-ml > "$O/build-pvgemm.log" 2>&1
prc=$?
wait $sp
src=$?
echo "build sgemm rc $src pvgemm rc $prc" | tee -a "$O/rc.txt"
for b in sgemm pvgemm; do
  grep -E "error|warning" "$O/build-$b.log" | head -40
  grep -E "spill|Used" "$O/build-$b.log" > "$O/regs-$b.txt"
done
grep -B2 -E "[1-9][0-9]* bytes spill" "$O/build-pvgemm.log" | grep -E "entry|spill" | head -20
[ "$src" = 0 ] && [ "$prc" = 0 ] || { tail -30 "$O/build-pvgemm.log"; tail -30 "$O/build-sgemm.log"; exit 4; }
[ "$STAGE" = build ] && exit 0

export REPS=${REPS:-7} PV_S=${PV_S:-"2048 8192"}
export SHAPES=${SHAPES:-"qkv:8192:6144:4096:1 o:8192:4096:4096:1 gate_up:8192:28672:4096:1 down:8192:4096:14336:1 sq4096:4096:4096:4096:1 qk2k:2048:2048:128:32 pv2k:2048:128:2048:32 qk8k:8192:8192:128:32 pv8k:8192:128:8192:32"}
ALS="pm1 u17 u17w u17p u17wp"
SG_DEF=plain_i8,plain_e4
for a in $ALS; do SG_DEF+=",gen_$a,sform_${a}_i8,slt_${a}_i8,slt_${a}_e4,sduo_${a}_i8"; done
export SG_ARMS=${SG_ARMS:-$SG_DEF}
mkdir -p "$B"/g{1,2,3,4} "$B"/v{1,2,3}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 12 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  { "$B/sgemm" gate "$B/g1" 256 256 512 1 0x5EED1234 3 && "$B/sgemm" gate "$B/g2" 256 128 128 2 7 2 &&
    "$B/sgemm" gate "$B/g3" 128 256 384 3 99 4 && "$B/sgemm" gate "$B/g4" 128 128 14336 1 5 1; } > "$O/gate-sgemm.txt" 2>&1
  echo "gate sgemm rc $?" >> "$O/rc.txt"
  { "$B/pvgemm" gate "$B/v1" 256 3 0x5EED1234 5 && "$B/pvgemm" gate "$B/v2" 384 2 7 3 &&
    "$B/pvgemm" gate "$B/v3" 128 4 99 2; } > "$O/gate-pvgemm.txt" 2>&1
  echo "gate pvgemm rc $?" >> "$O/rc.txt"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  # shellcheck disable=SC2086
  "$B/pvgemm" bench "$REPS" $PV_S > "$O/pvbench.jsonl" 2> "$O/pvbench.err"
  echo "bench pvgemm rc $?" >> "$O/rc.txt"
  # shellcheck disable=SC2086
  "$B/sgemm" bench "$REPS" $SHAPES > "$O/bench.jsonl" 2> "$O/bench.err"
  echo "bench sgemm rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/gate-sgemm.txt" "$O/gate-pvgemm.txt"
for g in v1 v2 v3; do python3 $D/pvref.py "$B/$g" >> "$O/pvgate.jsonl" 2>> "$O/pvgate.err"; echo "pvref $g rc $?" >> "$O/rc.txt"; done
for g in g1 g2 g3; do python3 $D/sref.py "$B/$g" >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "sref $g rc $?" >> "$O/rc.txt"; done
python3 $D/sref.py "$B/g4" --no-receipts >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "sref g4 rc $?" >> "$O/rc.txt"
python3 -c 'import json,sys
for f in sys.argv[1:]:
  for l in open(f): j = json.loads(l); print(j["dir"], "ok" if j["ok"] else "FAILED", j["n_checks"], "checks; failed:", j["failed"][:40], "unchecked:", j["unchecked_arms"])' \
  "$O/pvgate.jsonl" "$O/gate.jsonl"
tail -20 "$O/pvgate.err" "$O/gate.err"
python3 $D/pvtable.py "$O/pvbench.jsonl" "$O/pvgate.jsonl" > "$O/pvtable.md" 2> "$O/pvtable.err"
python3 $D/stable.py "$O/bench.jsonl" "$O/gate.jsonl" > "$O/table.md" 2> "$O/table.err"
cat "$O/rc.txt" "$O/pvtable.md" "$O/pvtable.err" "$O/table.md" "$O/table.err"
