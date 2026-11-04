#!/usr/bin/env bash
# hash_prof.py on one H200 in one short gpu-lease: the served Commit's rows-sha512 hashing at T 8,192, 16,384 and 32,768 in each
# VARIANT, then one variant under Nsight Systems (per-kernel time) and, with NCU=1, one 256 MiB window's kernels under Nsight Compute
# (occupancy, issue, stalls), the SM clock sampled beside it all. The extension is JIT-built before the lease. Measurement only.
# cwd: the source tree's root.
#   VENV (default /workspace/cache/kernels-kfuse/venv312), CUDA_HOME (/usr/local/cuda-13.0), BUILD (the JIT parent dir),
#   VARIANTS (default fold window fold16 fold256 nolevels phases1 fused), NSYS_VARIANT (fold), NCU (1), LEASE_MIN (15)
#   fold: the worker's levels kept, held leaves folded at ROWS_FOLD_BYTES, copies on a copy stream; window: each window's subtree
#   alone with its copies and one sync (the worker before the held folds); fold16, fold256: folds at 16 or 256 MiB of leaves;
#   nolevels: no levels kept; phases1, fused: the fold form with VERITY_ROWS_PHASES=1 or VERITY_ROWS_FORM=fused.
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

# a variant -> "KNOB_ENV|FLAGS": the kernels' own knobs as environment (read by the extension), hash_prof.py's flags
variant() {
  case $1 in
    fold) echo "|--form fold" ;;
    window) echo "|--form window" ;;
    fold16) echo "|--form fold --fold-mb 16" ;;
    fold256) echo "|--form fold --fold-mb 256" ;;
    nolevels) echo "|--levels 0" ;;
    phases1) echo "VERITY_ROWS_PHASES=1|--form fold" ;;
    fused) echo "VERITY_ROWS_FORM=fused|--form fold" ;;
    *) return 1 ;;
  esac
}

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,temperature.gpu --format=csv -lms 200 -i "${CUDA_VISIBLE_DEVICES:-0}" \
  > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
for v in ${VARIANTS:-fold window fold16 fold256 nolevels phases1 fused}; do
  spec=$(variant "$v") || { say "variant $v unknown"; continue; }
  ve=${spec%%|*}; va=${spec#*|}
  # shellcheck disable=SC2086
  env $ve python "$PROG" --out "$OUT" --label "$v" --knobs "$ve" $va > "$OUT/run.$v.log" 2>&1
  say "variant $v rc $?: $(grep -c '"T"' "$OUT/run.$v.log") steps"
done
NSYS=$CUDA_HOME/bin/nsys
nv=${NSYS_VARIANT:-fold}
if [ -x "$NSYS" ] && spec=$(variant "$nv"); then
  ve=${spec%%|*}; va=${spec#*|}
  # shellcheck disable=SC2086
  env $ve "$NSYS" profile -t cuda,nvtx -s none --cpuctxsw=none -f true -o "$OUT/$nv" python "$PROG" --out "$OUT" --label "nsys-$nv" \
    --reps 1 --knobs "$ve" $va > "$OUT/run.nsys.log" 2>&1
  say "nsys $nv rc $?"
  for r in cuda_gpu_kern_sum cuda_gpu_mem_time_sum nvtx_sum; do
    "$NSYS" stats --report "$r" --format csv -o "$OUT/nsys" "$OUT/$nv.nsys-rep" > /dev/null 2>&1
  done
  ls "$OUT"/nsys_* 2>/dev/null | tee -a "$OUT/steps.txt"
fi
NCU_BIN=$CUDA_HOME/bin/ncu
if [ "${NCU:-1}" = 1 ] && [ -x "$NCU_BIN" ]; then
  # one window of 65,536 rows (width 65,536, no pad window), in the window form: its salt and data kernels and the window's levels
  "$NCU_BIN" --kernel-name regex:rows_ --launch-count 12 \
    --section SpeedOfLight --section Occupancy --section LaunchStats --section SchedulerStats --section WarpStateStats \
    --section ComputeWorkloadAnalysis -f -o "$OUT/window" python "$PROG" --out "$OUT" --label ncu --no-check --form window \
    --tokens 256 --bytes-per-token 1048576 --reps 1 > "$OUT/run.ncu.log" 2>&1
  say "ncu rc $?"
  "$NCU_BIN" --import "$OUT/window.ncu-rep" --page details --csv > "$OUT/ncu_details.csv" 2> "$OUT/ncu_import.log"
  say "ncu import rc $? ($(wc -l < "$OUT/ncu_details.csv") lines)"
fi
