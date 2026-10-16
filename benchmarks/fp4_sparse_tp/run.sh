#!/usr/bin/env bash
# tp.cu on one GPU (cwd = the source tree): the three-piece 2:4-sparse NVFP4 route to P = X·W for X uniform on {±1..±k},
# gated bit for bit against a CPU int64 reference and timed against cuBLASLt int8 and FP8.  The three-piece arithmetic is
# gated with dense NVFP4 GEMMs; the sparse NVFP4 speed is probed via cuSPARSELt.  Build outside the lease.  Clocks read only.
# SIZES / KS / ROUNDS override the defaults.  Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source \
#     --env GPU_LEASE_WHO=HYB -- bash benchmarks/fp4_sparse_tp/run.sh
set -uxo pipefail
export O=${OUT:?}
export CUDA=/usr/local/cuda-13.0
export PATH=$CUDA/bin:$PATH
export SIZES=${SIZES:-1024 4096 8192} KS=${KS:-8 32 64} ROUNDS=${ROUNDS:-8}
D=benchmarks/fp4_sparse_tp

INC=$(find / -path '*nvidia/cusparselt/include/cusparseLt.h' 2>/dev/null | head -1)
INC=${INC:-$(find / -name cusparseLt.h 2>/dev/null | head -1)}
INCDIR=$(dirname "${INC:-/dev/null}")
LTSO=$(find / -name 'libcusparseLt.so*' 2>/dev/null | grep -E '0\.9|0\.8' | head -1)
LTSO=${LTSO:-$(find / -name 'libcusparseLt.so*' 2>/dev/null | head -1)}
LTDIR=$(dirname "${LTSO:-/dev/null}")
echo "INCDIR=$INCDIR LTDIR=$LTDIR LTSO=$LTSO" | tee "$O/libs.txt"
LINK=$(mktemp -d /tmp/ltlink.XXXXXX); ln -sf "$LTSO" "$LINK/libcusparseLt.so"

export B; B=$(mktemp -d /tmp/tp.XXXXXX); trap 'rm -rf "$B" "$LINK"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -o "$B/tp" $D/tp.cu \
  -I"$INCDIR" -I$CUDA/include -L"$LINK" -L"$LTDIR" -L$CUDA/lib64 -L$CUDA/lib64/stubs \
  -lcublasLt -lcusparseLt -lcusparse -lnvidia-ml > "$O/build.log" 2>&1 || { tail -60 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1

export LD_LIBRARY_PATH="$LTDIR:$CUDA/lib64:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-40}" -- bash -c '
  export LD_LIBRARY_PATH="'"$LTDIR"':'"$CUDA"'/lib64:${LD_LIBRARY_PATH:-}"
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics --format=csv,noheader -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  rc=0
  for n in $SIZES; do for k in $KS; do
    "'"$B"'/tp" "$n" "$k" 0x5EED1234 "$ROUNDS" >> "$O/tp.jsonl" 2>> "$O/errors.txt" || { echo "rc $? n=$n k=$k" >> "$O/errors.txt"; rc=1; }
  done; done
  nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks_event_reasons.active --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  exit $rc
'
rc=$?
cat "$O/tp.jsonl" 2>/dev/null
echo "=== errors ==="; cat "$O/errors.txt" 2>/dev/null
exit $rc
