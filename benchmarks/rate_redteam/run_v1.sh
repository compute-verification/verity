#!/usr/bin/env bash
# co.cu's sweep with the compression in V0 (adds on the ALU pipe) and V1 (adds as IMADs on the FMA pipe), in both squaring
# forms, against the encode and the decode at k = 1 … 3 (cwd = the source tree). Builds outside the lease; then, under one
# gpu-lease 1 --wait --max-min MAXMIN (default 15), the gate of every binary and the sweeps. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-rate-redteam -- taskset -c 48-95 bash benchmarks/rate_redteam/run_v1.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_redteam
PY=$(command -v python3 || echo /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python)
export PY
export B; B=$(mktemp -d /tmp/rate-redteam.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v)
P=()
for v in 0 1; do for sq in 0 1; do
  nvcc "${FLAGS[@]}" -DRT_B3V=$v -DWC_SQR=$sq -o "$B/co-v$v-$sq" $D/co.cu > "$O/co-v$v-$sq.ptxas" 2>&1 & P+=($!)
done; done
for p in "${P[@]}"; do wait "$p" || { tail -40 "$O"/co-v*.ptxas; exit 4; }; done
for v in 0 1; do
  cuobjdump -sass "$B/co-v$v-0" | awk '/Function : /{p = ($0 ~ /co_kernelILi1ELi1E/)} p' > "$O/co-v$v-encode-k1.sass"
  printf 'v%s IMAD %s IADD3 %s LOP3 %s SHF %s PRMT %s\n' $v \
    "$(grep -c 'IMAD' "$O/co-v$v-encode-k1.sass")" "$(grep -c 'IADD3' "$O/co-v$v-encode-k1.sass")" \
    "$(grep -c 'LOP3' "$O/co-v$v-encode-k1.sass")" "$(grep -c 'SHF' "$O/co-v$v-encode-k1.sass")" \
    "$(grep -c 'PRMT' "$O/co-v$v-encode-k1.sass")" >> "$O/sass-mix.txt"
done
sha256sum "$B"/co-* > "$O/bins.sha256"
for k in 1 2 3; do "$PY" $D/gate.py key --k $k --out "$B/key$k.txt" || exit 5; done
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-15}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,temperature.gpu \
          --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  rc=0
  for v in 0 1; do for sq in 0 1; do for what in 1 0; do for k in 1 2 3; do
    "$B/co-v$v-$sq" gate $what $k "$B/key$k.txt" "$B/gate.txt" >> "$O/gate.log" 2>&1 || rc=2
    "$PY" $D/gate.py check --what $what --k $k --key "$B/key$k.txt" --got "$B/gate.txt" >> "$O/gate.log" 2>&1 || rc=2
  done; done; done; done
  [ $rc = 0 ] || { echo "gate failed"; exit $rc; }
  for v in 0 1; do for sq in 0 1; do for what in 1 0; do for k in 1 2 3; do
    echo "== b3v $v sqr $sq what $what k $k" >> "$O/sweep.log"
    "$B/co-v$v-$sq" sweep $what $k "$B/key$k.txt" "$O/co-v$v-sq$sq-w$what-k$k.json" >> "$O/sweep.log" 2>&1 || rc=3
  done; q; done; done; done
  exit $rc'
