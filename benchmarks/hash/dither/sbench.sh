#!/usr/bin/env bash
# Dither's served standard number on node 2 (sbench.py), on one leased GPU for at most SBENCH_MAX_MIN (40) minutes: libh1c.so, libh1f.so
# and libh1o.so built for sm_120a first (the dither arms').  Extra arguments go to sbench.py (e.g. --arms exact,dither --gate --ops decode
# --t 64); SBENCH_SCRIPT runs another script with the same libraries and environment (h1ogate.py: the units kernels at
# kernel level).  DITHER_NSYS=1 runs it under nsys, recording the kernels inside the gate's CUDA profiler range (each graph's
# kernels as nodes), then the dither worker's kernels inside the gate's NVTX range (kernels.py over the SQLite export).
#   research run --on vy-nebius-2 --project verity --campaign dither-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-dither-rtx -- bash benchmarks/hash/dither/sbench.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
mkdir -p /workspace/pouw
S=$(mktemp -d /workspace/pouw/tmp-dither-sbench.XXXXXX); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD | tee "$OUT/commit.txt"
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$CUDA/bin:$PATH
for lib in h1c h1f h1o; do
  nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -Xcompiler -fPIC -shared \
    -o "$S/lib$lib.so" "benchmarks/hash/dither/$lib.cu" > "$OUT/build-$lib.log" 2>&1 \
    || { echo "sbench: NVCC_FAIL $lib"; grep -v "^ptxas info" "$OUT/build-$lib.log" | tail -30; exit 11; }
done
grep -B1 -A2 "h1f_k\|h1r_units" "$OUT/build-h1f.log" | grep -i "registers\|spill" | head -6
grep -B1 -A2 "h1o_units" "$OUT/build-h1o.log" | grep -i "registers\|spill" | head -8
grep -B1 -A2 "h1c_a_" "$OUT/build-h1c.log" | grep -i "registers\|spill" | head -6
[ -n "${BUILD_ONLY:-}" ] && { echo "sbench: built (BUILD_ONLY)"; exit 0; }
export DITHER_H1C=$S/libh1c.so DITHER_H1F=$S/libh1f.so DITHER_H1O=$S/libh1o.so VLLM_DISABLE_COMPILE_CACHE=1
# CA's credit gate (#2684) at its pinned head, for the dither-1s-credit arm
CREDIT_GATE_SHA=${CREDIT_GATE_SHA:-b25b34d96e}
git show "$CREDIT_GATE_SHA:integrations/vllm/verity_vllm/protocol_options/credit_gate.py" > "$S/credit_gate.py" 2>/dev/null \
  && export DITHER_CREDIT_GATE=$S/credit_gate.py || echo "sbench: no credit_gate.py at $CREDIT_GATE_SHA"
# CA's H1V3 (cursor/h1-v3-1172, with the wrap correction) at its pinned head: dither_one's gate runs its h1tile.reference
H1V3_SHA=${H1V3_SHA:-bd702a2c46}
mkdir -p "$S/h1v3" && git archive "$H1V3_SHA" verity 2>/dev/null | tar -x -C "$S/h1v3" && [ -f "$S/h1v3/verity/core/protocols/pouw/circuit/h1tile.py" ] \
  && export DITHER_H1V3_TREE=$S/h1v3 DITHER_H1V3_SHA=$H1V3_SHA || echo "sbench: no H1V3 tree at $H1V3_SHA"
# ASK-64's measured round trips (percall_rtt.py's percall-wait-ns.txt, sent with the run) for the dither-1s-percall-rtt arm
[ -f "$D/inputs/percall-wait-ns.txt" ] && export DITHER_PERCALL_WAITS=$D/inputs/percall-wait-ns.txt
cd integrations/vllm
PY=${PY:-/workspace/jobs/venv312/bin/python3}
export PATH=$(dirname "$PY"):$PATH
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/benchmarks/pouw/served_h1:$ROOT/benchmarks/hash/dither
export HF_HOME=${HF_HOME:-/workspace/hf} FLASHINFER_CUDA_ARCH_LIST=12.0f
# the wrap correction's pieces against H1V3 on the CPU, before the lease
"$PY" "$ROOT/benchmarks/hash/dither/wrapcheck.py" | tee "$OUT/wrapcheck.json"
case ${PIPESTATUS[0]} in 0) ;; 2) echo "sbench: wrapcheck skipped (no H1V3 tree)" ;; *) echo "sbench: WRAPCHECK_FAIL"; exit 12 ;; esac
NSYS=()
if [ "${DITHER_NSYS:-}" = 1 ]; then
  NSYS=("$CUDA/bin/nsys" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/gate" --force-overwrite true)
fi
echo "sbench: asking for one GPU at $(date -u +%H:%M:%SZ)"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 2 > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
MAX_MIN=${SBENCH_MAX_MIN:-40}
gpu-lease 1 --wait --max-min "$MAX_MIN" -- timeout $((60 * MAX_MIN - 50)) "${NSYS[@]}" "$PY" \
  "$ROOT/benchmarks/hash/dither/${SBENCH_SCRIPT:-sbench.py}" --out "$OUT" --scratch "$S" "$@"
rc=$?
kill "$SMI" 2>/dev/null
echo "sbench: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/gate.nsys-rep" ]; then
  "$CUDA/bin/nsys" stats --report cuda_gpu_kern_sum --format csv -o "$OUT/gate" "$S/gate.nsys-rep" > "$OUT/nsys-stats.log" 2>&1 \
    || echo "sbench: nsys stats rc $?"
  "$PY" "$ROOT/benchmarks/hash/dither/kernels.py" "$S/gate.sqlite" | tee "$OUT/kernels.json"
fi
exit "$rc"
