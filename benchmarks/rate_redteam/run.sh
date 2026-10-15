#!/usr/bin/env bash
# The red team's co-residency probe on one GPU (cwd = the source tree). Builds co.cu in both squaring forms outside the
# lease and writes the keys; then, under one gpu-lease 1 --wait --max-min MAXMIN (default 20), the gate (every codec
# chain against codec.py, every compression chain against its host build) and the sweeps. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/run.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
PY=$(command -v python3 || echo /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python)
export PY
export B; B=$(mktemp -d /tmp/rate-redteam.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v)
nvcc "${FLAGS[@]}" -DWC_SQR=0 -o "$B/co-0" $D/co.cu > "$O/co-0.ptxas" 2>&1 & P0=$!
nvcc "${FLAGS[@]}" -DWC_SQR=1 -o "$B/co-1" $D/co.cu > "$O/co-1.ptxas" 2>&1 & P1=$!
wait $P0 || { tail -40 "$O/co-0.ptxas"; exit 4; }
wait $P1 || { tail -40 "$O/co-1.ptxas"; exit 4; }
grep -A2 co_kernel "$O/co-0.ptxas" | grep -E "registers|spill" > "$O/regs.txt"
cuobjdump -sass "$B/co-0" | awk '/Function : /{p = ($0 ~ /co_kernelILi1ELi1E/)} p' > "$O/co-0-encode-k1.sass"
sha256sum "$B"/co-* > "$O/bins.sha256"
for k in 1 2 3; do "$PY" $D/gate.py key --k $k --out "$B/key$k.txt" || exit 5; done
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-20}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,temperature.gpu \
          --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  rc=0
  for sq in 0 1; do for what in 1 0; do for k in 1 2 3; do
    "$B/co-$sq" gate $what $k "$B/key$k.txt" "$B/gate-$sq-$what-$k.txt" >> "$O/gate.log" 2>&1 || rc=2
    "$PY" $D/gate.py check --what $what --k $k --key "$B/key$k.txt" --got "$B/gate-$sq-$what-$k.txt" >> "$O/gate.log" 2>&1 || rc=2
  done; done; done
  [ $rc = 0 ] || { echo "gate failed"; exit $rc; }
  for sq in 0 1; do for what in 1 0; do for k in 1 2 3; do
    echo "== sqr $sq what $what k $k" >> "$O/sweep.log"
    "$B/co-$sq" sweep $what $k "$B/key$k.txt" "$O/co-sq$sq-w$what-k$k.json" >> "$O/sweep.log" 2>&1 || rc=3
  done; q; done; done
  exit $rc'
