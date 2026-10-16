#!/usr/bin/env bash
# cuSPARSELt 2:4-sparse NVFP4 smoke test on one GPU (cwd = the source tree).  Locates cusparseLt.h and libcusparseLt.so on
# the node, builds outside the lease, runs a small CPU-checked sparse FP4 GEMM.  Clocks read only.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source \
#     --env GPU_LEASE_WHO=HYB -- bash benchmarks/fp4_sparse_tp/run_smoke.sh
set -uxo pipefail
export O=${OUT:?}
export CUDA=/usr/local/cuda-13.0
export PATH=$CUDA/bin:$PATH
D=benchmarks/fp4_sparse_tp

# cusparseLt header and shared object: prefer a pip-installed wheel (header+so matched), else the system .so.
INC=$(find / -path '*nvidia/cusparselt/include/cusparseLt.h' 2>/dev/null | head -1)
INC=${INC:-$(find / -name cusparseLt.h 2>/dev/null | head -1)}
INCDIR=$(dirname "${INC:-/dev/null}")
LTSO=$(find / -name 'libcusparseLt.so*' 2>/dev/null | grep -E '0\.9|0\.8' | head -1)
LTSO=${LTSO:-$(find / -name 'libcusparseLt.so*' 2>/dev/null | head -1)}
LTDIR=$(dirname "${LTSO:-/dev/null}")
echo "INCDIR=$INCDIR LTDIR=$LTDIR LTSO=$LTSO" | tee "$O/libs.txt"
ls -l "$LTDIR" >> "$O/libs.txt" 2>&1 || true
# Ensure a plain libcusparseLt.so symlink for -lcusparseLt.
LINK=$(mktemp -d /tmp/ltlink.XXXXXX)
ln -sf "$LTSO" "$LINK/libcusparseLt.so"

export B; B=$(mktemp -d /tmp/sp-smoke.XXXXXX); trap 'rm -rf "$B" "$LINK"' EXIT
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -o "$B/smoke" $D/sp_smoke.cu \
  -I"$INCDIR" -I$CUDA/include -L"$LINK" -L"$LTDIR" -L$CUDA/lib64 -lcusparseLt -lcusparse \
  > "$O/build.log" 2>&1 || { tail -60 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
git rev-parse HEAD > "$O/commit.txt" 2>&1

export LD_LIBRARY_PATH="$LTDIR:$CUDA/lib64:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-12}" -- bash -c '
  export LD_LIBRARY_PATH="'"$LTDIR"':'"$CUDA"'/lib64:${LD_LIBRARY_PATH:-}"
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm --format=csv,noheader -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "'"$B"'/smoke" 256 0x5EED1234 > "$O/smoke.jsonl" 2>> "$O/errors.txt"
'
rc=$?
cat "$O/smoke.jsonl" 2>/dev/null
cat "$O/errors.txt" 2>/dev/null
exit $rc
