#!/usr/bin/env bash
# The red team's read-and-rebuild probe on one GPU (cwd = the source tree). Builds rd.cu in both squaring forms outside
# the lease and writes the keys; then, under one gpu-lease 1 --wait --max-min MAXMIN (default 15), the gate (every codec
# chain against codec.py, every read thread's XOR against the host's) and the sweeps. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/rd.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
PY=$(command -v python3 || echo /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python)
export PY
export B; B=$(mktemp -d /tmp/rate-redteam-rd.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v)
nvcc "${FLAGS[@]}" -DWC_SQR=0 -o "$B/rd-0" $D/rd.cu > "$O/rd-0.ptxas" 2>&1 & P0=$!
nvcc "${FLAGS[@]}" -DWC_SQR=1 -o "$B/rd-1" $D/rd.cu > "$O/rd-1.ptxas" 2>&1 & P1=$!
wait $P0 || { tail -40 "$O/rd-0.ptxas"; exit 4; }
wait $P1 || { tail -40 "$O/rd-1.ptxas"; exit 4; }
sha256sum "$B"/rd-* > "$O/bins.sha256"
for k in 1 2 3; do "$PY" $D/gate.py key --k $k --out "$B/key$k.txt" || exit 5; done
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-15}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
          --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  rc=0
  for sq in 0 1; do for k in 1 2 3; do
    "$B/rd-$sq" gate $k "$B/key$k.txt" "$B/gate-$sq-$k.txt" >> "$O/gate.log" 2>&1 || rc=2
    "$PY" $D/gate.py check --what 1 --k $k --key "$B/key$k.txt" --got "$B/gate-$sq-$k.txt" >> "$O/gate.log" 2>&1 || rc=2
  done; done
  [ $rc = 0 ] || { echo "gate failed"; exit $rc; }
  for sq in 0 1; do for k in 1 2 3; do
    echo "== sqr $sq k $k" >> "$O/sweep.log"
    "$B/rd-$sq" sweep $k "$B/key$k.txt" "$O/rd-sq$sq-k$k.json" 16384 >> "$O/sweep.log" 2>&1 || rc=3
  done; q; done
  exit $rc'
