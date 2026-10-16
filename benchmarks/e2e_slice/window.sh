#!/usr/bin/env bash
# The slice's window on vy-nebius-2 (cwd = the source tree's root): C-Flock's CPU build and the venv by build.sh (cached),
# the engine by nvcc, then under one GPU lease the challenger (its own process, its own circuit) and the prover for ROUNDS
# rounds of CALLS o_proj calls each, a coin every CADENCE seconds at the soonest; after the window, the operand
# commitment's GPU cost kernel on the same GPU (slice commit). Clocks are read, never set. Under $OUT:
# challenger/{rounds.jsonl,window.json}, prover/{rounds.jsonl,prover.json}, gpu.txt, smi.csv, commit.jsonl, summary.json.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=E2E \
#     --env ROUNDS=24 -- bash benchmarks/e2e_slice/window.sh
# ROUNDS (24), CALLS (4), CADENCE (10), MAXMIN (the lease, 30), PROVER_ARGS, CHALLENGER_ARGS.
set -uo pipefail
export O=${OUT:?} D=benchmarks/e2e_slice
export ROUNDS=${ROUNDS:-24} CALLS=${CALLS:-4} CADENCE=${CADENCE:-10} MAXMIN=${MAXMIN:-30}
git rev-parse HEAD > "$O/commit.txt" 2>&1
bash $D/build.sh > "$O/build.log" 2>&1 || { tail -20 "$O/build.log"; exit 2; }
PY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["python"])' "$O/builds.json")
FLOCK=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prove"])' "$O/builds.json")
export PY FLOCK
export PATH=/usr/local/cuda-13.0/bin:$PATH
S=$(mktemp -d /dev/shm/e2e-window.XXXXXX)
export S
trap 'rm -rf "$S"' EXIT
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o "$S/slice" $D/slice.cu -lcuda -lcublasLt -lnvidia-ml \
  > "$O/build-slice.log" 2>&1 || { tail -40 "$O/build-slice.log"; exit 4; }
PYTHONPATH=$PWD:$PWD/verity/ml/flock/python:$PWD/integrations/vllm:$PWD/tools/verity/circuit_check/src
export PYTHONPATH
{ nproc; lscpu | grep -E 'Model name'; grep MemTotal /proc/meminfo; } > "$O/host.txt"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "$MAXMIN" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit \
    --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 1000 \
    -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  $PY $D/challenger.py --flock "$FLOCK" --out "$O/challenger" --scratch "$S/c" --port-file "$S/port" --cadence "$CADENCE" \
    ${CHALLENGER_ARGS:-} > "$O/challenger.log" 2>&1 &
  ch=$!
  until [ -s "$S/port" ]; do kill -0 $ch 2>/dev/null || break; sleep 1; done
  if [ -s "$S/port" ]; then
    $PY $D/prover.py --engine "$S/slice" --flock "$FLOCK" --port "$(cat "$S/port")" --out "$O/prover" --scratch "$S/p" \
      --rounds "$ROUNDS" --calls "$CALLS" ${PROVER_ARGS:-} > "$O/prover.log" 2>&1
    echo "prover rc $?" >> "$O/rc.txt"
  fi
  wait $ch; echo "challenger rc $?" >> "$O/rc.txt"
  "$S/slice" commit 7 8192 4096 > "$O/commit.jsonl" 2> "$O/commit.err"; echo "commit rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -3 "$O/prover.log" "$O/challenger.log" 2>/dev/null
$PY $D/window_summary.py "$O" > "$O/summary.json"
cat "$O/summary.json"
