#!/usr/bin/env bash
# v31's own fused de-noise beside denoise.cu's cuBLASLt estimate: `v31 bench` (v3.1's call, its peel as the GEMM's tail stages)
# on Llama-3.2-1B's linears with V31_DIAG=1, so gemm - gemm_notail is the peel, gemm - gemm_nost the T_s stores, and lt
# cuBLASLt's plain GEMM. v31 needs m, N, K and L' multiples of 128, so m 512 and 8,192 at L' 128 and 512. Timing only: the
# bench's checked call and its sampled tiles are dumped but not judged here (run.sh's gates and gpu-sample judge v31).
#
#   research run --on vy-dc-h200-5 --project verity --campaign CAMPAIGN --source . --cwd source --as LANE \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/denoise/v31_peel.sh
#
# Env: MS (default "512 8192"), CONFIGS (L':t, default "128:8 512:2"), ROW (default 16384), REPS (default 5), LEASE.
set -uo pipefail
D=benchmarks/pouw/accel_v31
O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH PYTHONPATH=.
PY=${PY:-uv run -q --no-project --with numpy python}
export O ROW=${ROW:-16384}
nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 \
  -o "$O/v31" $D/v31.cu -lcuda -lcublasLt > "$O/build.log" 2>&1 || { tail -30 "$O/build.log"; exit 3; }
declare -A NK=([qkv]=3072:2048 [o]=2048:2048 [gu]=16384:2048 [down]=2048:8192)
shapes=()
for m in ${MS:-512 8192}; do
  for l in qkv o gu down; do shapes+=("$l-m$m:$m:${NK[$l]}"); done
done
: > "$O/rc.txt"
for c in ${CONFIGS:-128:8 512:2}; do
  $PY $D/v31.py params "$O/params" "${shapes[@]}" --lp "${c%%:*}" --t "${c##*:}" --row "$ROW" 2>> "$O/prep.err"
  echo "params $c rc $?" >> "$O/rc.txt"
done
# shellcheck disable=SC2016
${LEASE-gpu-lease 1 --wait --max-min 40 --} bash -c '
  for c in ${CONFIGS:-128:8 512:2}; do
    lp=${c%%:*} t=${c##*:}
    mkdir -p "$O/lp$lp-t$t"
    V31_DIAG=1 V31_CPRIO=lo "$O/v31" bench "${REPS:-5}" "$O/lp$lp-t$t" "$O/params" "$lp" "$t" "$ROW" "$@" \
      > "$O/lp$lp-t$t/bench.jsonl" 2> "$O/lp$lp-t$t/bench.err"
    echo "bench $c rc $?" >> "$O/rc.txt"
  done' _ "${shapes[@]}"
cat "$O/rc.txt"
! grep -qv ' rc 0$' "$O/rc.txt"
