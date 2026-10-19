#!/usr/bin/env bash
# The kernel-level standard number on one dc node: kb.so (and kb_pg.so on sm_100a) built outside the lease, kbench.py under
# one gpu-lease (gates before timing, then the interleaved arms), verify.py after it (CPU, CA's numpy twin).
#
#   research run --on vy-dc-h200 --project verity --campaign dither-loop --source . --cwd source --timeout 40m \
#     -- bash benchmarks/hash/dither/job.sh sm_90a
#
# Env: KB_ARGS (extra kbench.py flags, e.g. "--ms 1,32 --linears o"), LEASE (gpu-lease flags, default "--max-min 14"),
# VERITY_BENCH_VENV (default /workspace/cp/venv312), NVCC (default CUDA 13.0's).
# Exit: kbench's (4 a gate failed), 5 verify.py's twin disagreed, 2 a build failed.
set -uo pipefail
ARCH=${1:?usage: job.sh sm_90a|sm_100a}
HERE=$(cd "$(dirname "$0")" && pwd)
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
WORK=$(mktemp -d)
mkdir -p "$OUT/env"
"$NVCC" --version > "$OUT/env/nvcc.txt" 2>&1
LIB64=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)
NV=("$NVCC" -O3 -std=c++17 -shared -Xcompiler -fPIC -gencode "arch=compute_${ARCH#sm_},code=$ARCH")
"${NV[@]}" "$HERE/kb.cu" -lcublasLt -Xlinker "-rpath=$LIB64" -o "$WORK/kb.so" 2> "$OUT/env/build-kb.log" \
  || { cat "$OUT/env/build-kb.log"; exit 2; }
PG=""
if [ "$ARCH" = sm_100a ]; then
  "${NV[@]}" "$HERE/kb_pg.cu" -o "$WORK/kb_pg.so" 2> "$OUT/env/build-kb_pg.log" || { cat "$OUT/env/build-kb_pg.log"; exit 2; }
  PG="--pg-lib $WORK/kb_pg.so"
fi
for so in "$WORK"/*.so; do "$(dirname "$NVCC")/cuobjdump" --dump-resource-usage "$so" > "$OUT/env/$(basename "$so").res.txt" 2>&1; done
export PYTHONPATH=$PWD/integrations/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH} KB_CMD="$PY $HERE/kbench.py --lib $WORK/kb.so $PG --out $OUT ${KB_ARGS:-}"
bench='
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  $KB_CMD 2>&1 | tee "$OUT/kbench.out"
  rc=${PIPESTATUS[0]}
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
  exit $rc'
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  bash -c "$bench"
else
  read -r -a lease <<< "${LEASE:---max-min 14}"
  GPU_LEASE_WHO=kernels-dither-hb gpu-lease 1 --wait "${lease[@]}" -- bash -c "$bench"
fi
rc=$?
echo "lease rc $rc"
"$PY" "$HERE/verify.py" "$OUT" 2>&1 | tail -3
vrc=${PIPESTATUS[0]}
exit $(( rc ? rc : vrc ))
