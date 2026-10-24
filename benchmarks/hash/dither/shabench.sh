#!/usr/bin/env bash
# shabench.cu built for SHA_ARCH (default sm_120a; sm_90a on vy-dc-h200), then shabench.py on one leased GPU for at most
# 10 minutes: SHA-512 compression variants checked against hashlib, then their latency (one warp) and throughput (every
# SM), into out/ARCH.  Extra arguments go to shabench.py.  SHA_BUILD_ONLY=1 stops after the build and each variant's SASS counts.
#   research run --on vy-nebius-2 --project verity --campaign dither-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-dither-rtx2 -- bash benchmarks/hash/dither/shabench.sh
set -uo pipefail
ARCH=${SHA_ARCH:-sm_120a}
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out/$ARCH; mkdir -p "$OUT"
S=$(mktemp -d "${TMPDIR:-/tmp}/dither-sha.XXXXXX"); trap 'rm -rf "$S"' EXIT
git rev-parse HEAD 2>/dev/null | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" -Xptxas -v -Xcompiler -fPIC -shared \
  -o "$S/libshabench.so" benchmarks/hash/dither/shabench.cu > "$OUT/build.log" 2>&1 \
  || { echo "shabench: NVCC_FAIL"; grep -v "^ptxas info" "$OUT/build.log" | tail -30; exit 11; }
grep -A2 "chain_k" "$OUT/build.log" | grep -i "registers\|spill" | head -18
"$CUDA/bin/cuobjdump" -sass "$S/libshabench.so" > "$OUT/sass.txt" 2>/dev/null
for v in 0 1 2 3 4 5 6 7 8; do
  awk "/Function : .*chain_kILi${v}E/{f=1;next} /Function :/{f=0} f" "$OUT/sass.txt" | grep -o '/\*[0-9a-f]*\*/ *[A-Z0-9]*' \
    | awk -v v="$v" '{n++; o=$2} o~/^(SHF|LOP3|IADD3|LEA|PRMT|ISETP|SEL|VIADD)$/{alu++} o~/^IMAD/{fma++}
      END {printf "V%s sass %d: alu-pipe %d, imad %d\n", v, n, alu, fma}'
done | tee "$OUT/sass-counts.txt"
[ -n "${SHA_BUILD_ONLY:-}" ] && exit 0
PY=${PY:-/workspace/jobs/venv312/bin/python3}
echo "shabench: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --max-min 10 -- timeout 550 "$PY" benchmarks/hash/dither/shabench.py --lib "$S/libshabench.so" --out "$OUT" "$@"
rc=$?
echo "shabench: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
