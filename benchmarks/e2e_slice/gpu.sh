#!/usr/bin/env bash
# The slice's GPU stage on one leased sm_120 GPU (cwd = the source tree's root): builds slice.cu outside the lease, then under
# one short lease the engine's gate (gate.py: P_A, three tiles' duo receipts and served product against hashlib and numpy, at
# a small shape and at o_proj), lpgemm.cu's interleaved timing of o_proj's routes (LP_ARMS) and the operand commitment's
# SHA-512 work per call (slice commit). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=E2E \
#     -- bash benchmarks/e2e_slice/gpu.sh
# STAGE=build|gate|all (default all; gate: the gates only); GATE_KINDS (default "duo duos"); REPS (default 7); LP_ARMS.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/e2e_slice
export BIN=${E2E_BIN:-/tmp/e2e-slice-$$}; mkdir -p "$BIN"
export PY=${E2E_PYTHON:-python3}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$BIN/slice" $D/slice.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build-slice.log" 2>&1
rc=$?
echo "build slice rc $rc" | tee -a "$O/rc.txt"
grep -E "error|warning" "$O/build-slice.log" | head -20
[ "$rc" = 0 ] || { tail -40 "$O/build-slice.log"; exit 4; }
[ "${STAGE:-all}" = build ] && exit 0
export REPS=${REPS:-7} GATE_KINDS=${GATE_KINDS:-duo duos} STAGE=${STAGE:-all}
export LP_ARMS=${LP_ARMS:-plain,bonly,td_bsep,td_bfus,fd_bsep,fd_bfus,rd_bsep,rd_bfus,t_bsep,lt_bsep}
nproc > "$O/nproc.txt"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 12 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  W=$(mktemp -d /dev/shm/e2e-gate.XXXXXX)
  for kd in $GATE_KINDS; do
    for s in "256 256 512" "8192 4096 4096"; do
      $PY $D/gate.py "$BIN/slice" "$W/g" $s $kd >> "$O/gate.jsonl" 2>> "$O/gate.err"; echo "gate $s $kd rc $?" >> "$O/rc.txt"
    done
  done
  rm -rf "$W"
  [ "$STAGE" = gate ] && exit 0
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$BIN/slice" bench "$REPS" o:8192:4096:4096:1 > "$O/bench.jsonl" 2> "$O/bench.err"; echo "bench rc $?" >> "$O/rc.txt"
  "$BIN/slice" commit "$REPS" 8192 4096 > "$O/commit.jsonl" 2> "$O/commit.err"; echo "commit rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
rm -rf "$BIN"
cat "$O/rc.txt"
$PY -c 'import json,sys
for l in open(sys.argv[1]): j = json.loads(l); print(j["m"], j["n"], j["k"], j.get("kind", "duo"), "ok" if j["ok"] else "FAILED", len(j["checks"]), "checks; failed:", j["failed"], "calls:", j["calls"])' "$O/gate.jsonl"
tail -5 "$O/gate.err"
[ "$STAGE" = gate ] || $PY $D/gpu_summary.py "$O/bench.jsonl" "$O/commit.jsonl"
