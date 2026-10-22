#!/usr/bin/env bash
# shabench.cu built for sm_120a, then shabench.py on one leased GPU for at most 10 minutes: SHA-512 compression variants
# checked against hashlib, then their latency (one warp) and throughput (every SM).  Extra arguments go to shabench.py.
#   research run --on vy-nebius-2 --project verity --campaign dither-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-dither-rtx2 -- bash benchmarks/hash/dither/shabench.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-dither-sha.XXXXXX); trap 'rm -rf "$S"' EXIT
git rev-parse HEAD | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -Xcompiler -fPIC -shared \
  -o "$S/libshabench.so" benchmarks/hash/dither/shabench.cu > "$OUT/build.log" 2>&1 \
  || { echo "shabench: NVCC_FAIL"; grep -v "^ptxas info" "$OUT/build.log" | tail -30; exit 11; }
grep -A2 "chain_k" "$OUT/build.log" | grep -i "registers\|spill" | head -6
"$CUDA/bin/cuobjdump" -sass "$S/libshabench.so" > "$OUT/sass.txt" 2>/dev/null
for v in 0 1 2; do
  echo "V$v sass lines: $(awk "/Function : .*chain_kILi${v}E/{f=1;next} /Function :/{f=0} f" "$OUT/sass.txt" | grep -c '/\*[0-9a-f]*\*/')"
done
PY=${PY:-/workspace/jobs/venv312/bin/python3}
echo "shabench: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --max-min 10 -- timeout 550 "$PY" benchmarks/hash/dither/shabench.py --lib "$S/libshabench.so" --out "$OUT" "$@"
rc=$?
echo "shabench: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
