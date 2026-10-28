#!/usr/bin/env bash
# NOBITS probe on one GPU (cwd = the source tree): does a kind::mxf4nvf4 MMA leave the committed s32 word's bits in a
# register for free on sm_120?  Builds nobits.cu, dumps its SASS (for the per-word conversion instruction count), and runs
# the probe under the lease.  Clocks read only.  NTILES / SEED override the defaults.  Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/fp4_nobits/run.sh
set -uxo pipefail
export O=${OUT:?}
export CUDA=/usr/local/cuda-13.0
export PATH=$CUDA/bin:$PATH
export NTILES=${NTILES:-256} SEED=${SEED:-0xB175C0DE}
D=benchmarks/pouw/fp4_nobits

export B; B=$(mktemp -d /tmp/nobits.XXXXXX); trap 'rm -rf "$B"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -o "$B/nobits" $D/nobits.cu \
  -I$CUDA/include -L$CUDA/lib64 -L$CUDA/lib64/stubs -lnvidia-ml > "$O/build.log" 2>&1 || { tail -60 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1

# SASS of the whole binary and of the per-word conversion / MMA kernels (no GPU needed).
cuobjdump -sass "$B/nobits" > "$O/sass.txt" 2>&1 || true
for fn in conv_f2i conv_binade s8_kernel nvfp4_kernel; do
  echo "===== $fn =====" >> "$O/sass_kernels.txt"
  cuobjdump -sass -fun "$(cuobjdump -symbols "$B/nobits" 2>/dev/null | awk -v f="$fn" '$NF ~ f {print $NF; exit}')" "$B/nobits" >> "$O/sass_kernels.txt" 2>&1 || true
done
# Instruction counts that answer "how many ops a word": F2I in conv_f2i, IADD3 with the binade immediate in conv_binade.
{
  echo "F2I_in_conv_f2i=$(grep -c 'F2I' "$O/sass.txt" 2>/dev/null || echo 0)"
  echo "IADD3_binade_imm=$(grep -Ec 'IADD3.*(4b400000|-0x4b400000|b4c00000)' "$O/sass.txt" 2>/dev/null || echo 0)"
} > "$O/sass_counts.txt" 2>&1 || true

gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm --format=csv,noheader -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "'"$B"'/nobits" "$NTILES" "$SEED" > "$O/nobits.jsonl" 2> "$O/errors.txt"
  rc=$?
  nvidia-smi --query-gpu=index,uuid,clocks.sm --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $rc
'
rc=$?
echo "=== nobits.jsonl ==="; cat "$O/nobits.jsonl" 2>/dev/null
echo "=== sass_counts ==="; cat "$O/sass_counts.txt" 2>/dev/null
echo "=== errors ==="; cat "$O/errors.txt" 2>/dev/null
exit $rc
