#!/usr/bin/env bash
# hybrid.cu on one GPU (cwd = the source tree): the FP4 core-plus-correction route to A·Bᵀ for the non-FP4-holdable
# alphabets u5, {±1,±2,±4,±5} and u8, checked bit for bit against a CPU int64 reference (every entry at n = 1024, sampled
# at 4096/8192 via int8 whose own output is CPU-checked), then timed interleaved against cuBLASLt int8 and FP8. The FP4
# core and the dense/sparse correction backends are also timed on their own for the split. Build outside the lease. Clocks
# read only.  SIZES / ALPHABETS override the defaults; ROUNDS is the number of interleaved rounds.  Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source \
#     --env GPU_LEASE_WHO=HYB -- bash benchmarks/d4_attacks/run_hybrid.sh
set -uxo pipefail
export O=${OUT:?}
export CUDA=/usr/local/cuda-13.0
export PATH=$CUDA/bin:$PATH
export SIZES=${SIZES:-1024 4096 8192} ALPHABETS=${ALPHABETS:-u5 s1245 u8} ROUNDS=${ROUNDS:-8}
export B; B=$(mktemp -d /tmp/hyb-attacks.XXXXXX); trap 'rm -rf "$B"' EXIT
D=benchmarks/d4_attacks
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/hybrid" $D/hybrid.cu \
  -L$CUDA/lib64 -L$CUDA/lib64/stubs -lcublasLt -lnvidia-ml > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
cuobjdump -sass "$B/hybrid" > "$B/hybrid.sass" 2>&1
grep -oE 'MUFU|\.FTZ' "$B/hybrid.sass" | sort | uniq -c > "$O/sass_flags.txt" || true
gzip -c "$B/hybrid.sass" > "$O/hybrid.sass.gz"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-28}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,temperature.gpu \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  rc=0
  for n in $SIZES; do
    for a in $ALPHABETS; do
      "$B/hybrid" "$n" "$a" 0x5EED1234 "$ROUNDS" >> "$O/hybrid.jsonl" 2>> "$O/errors.txt" || { echo "rc $? n=$n a=$a" >> "$O/errors.txt"; rc=1; }
    done
  done
  nvidia-smi --query-gpu=index,uuid,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $rc
'
rc=$?
python3 $D/table_hybrid.py "$O/hybrid.jsonl" > "$O/table.md" 2>> "$O/errors.txt"
cat "$O/table.md"
exit $rc
