#!/usr/bin/env bash
# register_lookup.cu on one GPU (cwd = the source tree): for each alphabet, cuBLASLt int8 and the standalone lookups
# whose tables aren't in shared memory (Four Russians from an L1-resident global table, PRMT register tables, SHFL warp
# tables, each with its split past the table limit), gated bit for bit (int8 against a CPU int64 reference, the rest
# against int8), then timed interleaved with the SM clock read around every item. Build outside the lease; clocks
# read only. Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source \
#     --env GPU_LEASE_WHO=LKP -- bash benchmarks/d4_attacks/run_register.sh
set -uxo pipefail
export O=${OUT:?}
export CUDA=/usr/local/cuda-13.0
export PATH=$CUDA/bin:$PATH
export SIZES=${SIZES:-1024 4096 8192} ALPHABETS=${ALPHABETS:-k2 k4 k6 k8 u6} ROUNDS=${ROUNDS:-8}
export B; B=$(mktemp -d /tmp/lkp-register.XXXXXX); trap 'rm -rf "$B"' EXIT
D=benchmarks/d4_attacks
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -o "$B/register_lookup" $D/register_lookup.cu \
  -L$CUDA/lib64 -L$CUDA/lib64/stubs -lcublasLt -lnvidia-ml > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1
cuobjdump -sass "$B/register_lookup" > "$B/register.sass" 2>&1
for k in k_fr_l1 k_prmtILb0 k_prmtILb1 k_shflILi1ELb0 k_shflILi2ELb0 k_shflILi2ELb1; do
  echo "== $k" >> "$O/opcodes.txt"
  awk -v pat="$k" '$0 ~ "Function : " {f = index($0, pat) > 0; next} f' "$B/register.sass" \
    | grep -oE '^\s+/\*[0-9a-f]+\*/\s+[A-Z][A-Z0-9_.]*' | awk '{print $2}' | sort | uniq -c | sort -rn | head -20 >> "$O/opcodes.txt"
done
gzip -c "$B/register.sass" > "$O/register.sass.gz"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,temperature.gpu \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  rc=0
  for n in $SIZES; do
    for a in $ALPHABETS; do
      "$B/register_lookup" "$n" "$a" 0x5EED1234 "$ROUNDS" >> "$O/register.jsonl" 2>> "$O/errors.txt" \
        || { echo "rc $? n=$n a=$a" >> "$O/errors.txt"; rc=1; }
    done
  done
  nvidia-smi --query-gpu=index,uuid,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $rc
'
rc=$?
grep '"event":"summary"' "$O/register.jsonl" > "$O/summary.jsonl" || true
exit $rc
