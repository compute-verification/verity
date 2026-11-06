#!/usr/bin/env bash
# h1c.cu built for sm_120a on the CPU, then h1cgate.py (or H1C_GATE, e.g. h1agate.py: the served calls' A tables) on one
# leased GPU for at most 25 minutes: the served window's tables on a full step against commit.py, then their timing.  Extra
# arguments go to the gate.
# On an RTX PRO 6000 under a GPU lease:
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/h1c.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-dither-h1c.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
"$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -Xcompiler -fPIC -shared \
  -o "$S/libh1c.so" benchmarks/hash/dither/h1c.cu > "$OUT/build-h1c.log" 2>&1 \
  || { echo "h1c: NVCC_FAIL"; grep -v "^ptxas info" "$OUT/build-h1c.log" | tail -30; exit 11; }
grep -A2 "h1c_" "$OUT/build-h1c.log" | grep -i "registers\|spill" | head
[ -n "${BUILD_ONLY:-}" ] && { echo "h1c: built (BUILD_ONLY)"; exit 0; }
[ -x /workspace/jobs/venv312/bin/python3 ] && PY=${PY:-/workspace/jobs/venv312/bin/python3}
PY=${PY:-$(. benchmarks/harness/env.sh && echo "$VENV/bin/python3")} || exit 2
export PYTHONPATH=$ROOT:$ROOT/tools/config:$ROOT/benchmarks/pouw/served_h1:$ROOT/benchmarks/hash/dither
echo "h1c: asking for one GPU at $(date -u +%H:%M:%SZ)"
gpu-lease 1 --wait --max-min 25 -- timeout 1450 "$PY" "benchmarks/hash/dither/${H1C_GATE:-h1cgate.py}" --lib "$S/libh1c.so" \
  --out "$OUT" "$@"
rc=$?
echo "h1c: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
