#!/usr/bin/env bash
# hash_prof.py on one H200 in one short gpu-lease: the served Commit's rows-sha512 hashing at T 8,192, 16,384 and 32,768 in its
# served form and with each of the kernels' knobs (VERITY_ROWS_PHASES=1, VERITY_ROWS_FORM=fused, no levels kept), then the served
# form under Nsight Systems (per-kernel time) and one 256 MiB window's kernels under Nsight Compute (occupancy, issue, stalls), the SM
# clock sampled beside it all. The extension is JIT-built before the lease. Measurement only. cwd: the source tree's root.
#   VENV (default /workspace/cache/kernels-kfuse/venv312), CUDA_HOME (/usr/local/cuda-13.0), BUILD (the JIT parent dir),
#   VARIANTS (served phases1 fused nolevels), LEASE_MIN (15)
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=${TREE:-$PWD}
VENV=${VENV:-/workspace/cache/kernels-kfuse/venv312}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm${PYTHONPATH:+:$PYTHONPATH}
export NATIVE_COLLECT_BUILD=${BUILD:-/workspace/research/served-commit-hash/nc} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export OUT TREE VENV
PROG=$TREE/benchmarks/served_commit/hash_prof.py
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  mkdir -p "$NATIVE_COLLECT_BUILD"
  research keep "$(dirname "$NATIVE_COLLECT_BUILD")" --owner @kernels --expires 3d --priority normal \
    --why "served_commit/hash_prof.sh: the collector's JIT builds, keyed by source digest" > /dev/null 2>&1
  say "build $(git -C "$TREE" rev-parse --short=10 HEAD 2>/dev/null) into $NATIVE_COLLECT_BUILD"
  python -c 'from verity_vllm.commit.committer.native_ext import ext; ext()' > "$OUT/build.log" 2>&1
  rc=$?; say "build rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-15}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  for f in "$OUT"/nsys_*.csv; do [ -f "$f" ] && { echo "== $f"; head -40 "$f"; }; done
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,temperature.gpu --format=csv -lms 200 -i "${CUDA_VISIBLE_DEVICES:-0}" \
  > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
for v in ${VARIANTS:-served phases1 fused nolevels}; do
  case $v in
    served) env LABEL=served python "$PROG" ;;
    phases1) env LABEL=phases1 VERITY_ROWS_PHASES=1 python "$PROG" ;;
    fused) env LABEL=fused VERITY_ROWS_FORM=fused python "$PROG" ;;
    nolevels) env LABEL=nolevels LEVELS=0 python "$PROG" ;;
    *) say "variant $v unknown"; continue ;;
  esac > "$OUT/run.$v.log" 2>&1
  say "variant $v rc $?: $(grep -c '"T"' "$OUT/run.$v.log") steps"
done
NSYS=$CUDA_HOME/bin/nsys
if [ -x "$NSYS" ]; then
  LABEL=nsys REPS=1 "$NSYS" profile -t cuda,nvtx -s none --cpuctxsw=none -f true -o "$OUT/served" python "$PROG" > "$OUT/run.nsys.log" 2>&1
  say "nsys rc $?"
  for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum nvtx_sum; do
    "$NSYS" stats --report "$r" --format csv -o "$OUT/nsys" "$OUT/served.nsys-rep" > /dev/null 2>&1
  done
  ls "$OUT"/nsys_* 2>/dev/null | tee -a "$OUT/steps.txt"
fi
NCU=$CUDA_HOME/bin/ncu
if [ -x "$NCU" ]; then
  # one window of 65,536 rows (width 65,536, no pad window): its salt and data kernels and the window's levels
  LABEL=ncu CHECK=0 TOKENS=256 BYTES_PER_TOKEN=1048576 REPS=1 "$NCU" --kernel-name regex:rows_ --launch-count 12 \
    --section SpeedOfLight --section Occupancy --section LaunchStats --section SchedulerStats --section WarpStateStats \
    --section ComputeWorkloadAnalysis -f -o "$OUT/window" python "$PROG" > "$OUT/run.ncu.log" 2>&1
  say "ncu rc $?"
  "$NCU" --import "$OUT/window.ncu-rep" --page details --csv > "$OUT/ncu_details.csv" 2> "$OUT/ncu_import.log"
  say "ncu import rc $? ($(wc -l < "$OUT/ncu_details.csv") lines)"
fi
