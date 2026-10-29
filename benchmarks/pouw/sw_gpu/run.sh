#!/usr/bin/env bash
# sw_gpu's measurements on one H200 in one gpu-lease: the binaries are built and their SASS gated (sass_loops.py, pins.json)
# before the lease; under it, each binary's correctness check runs before its timing, the SM clock sampled beside it all.
# Measurement only. cwd: the source tree's root.
#   WHAT     what to run, joined by + (default bench+k32): bench (sw_bench: forming and coins, whole calls), k32 (k32: wgmma
#            with every k32 word leaving the accumulator, and forming beside it) and dgemm (CORE-H32's ../dc/dgemm.cu: a whole
#            GEMM with every k32 word folded by duo, without, and cuBLASLt, after its three exact gates)
#   DGEMM_SHAPES  dgemm's bench shapes, M,N,K each (default: Llama-3.2-1B's linears at m 512 and 8,192, the
#            DeepSeek-V4-like shapes at 8,192, and 8,192 cubed)
#   SIZES    sw_bench's operand sizes in entries (default: every per-call X and W of Llama-3.2-1B and the DeepSeek-V4-like shapes
#            at m 64, 512 and 8,192, and 2^30)
#   REPS (7), STEPS (k32's wgmma steps a CTA, 65536), CUDA_HOME (/usr/local/cuda-13.0), LEASE_MIN (20)
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }
HERE=benchmarks/pouw/sw_gpu
WHAT=${WHAT:-bench+k32}
has() { [[ "+$WHAT+" == *"+$1+"* ]]; }

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/sw-gpu-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  NVCC=${CUDA_HOME:-/usr/local/cuda-13.0}/bin/nvcc
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) ($WHAT) into $BUILD_DIR"
  for b in bench:sw_bench k32:k32; do
    has "${b%%:*}" || continue
    bin=${b#*:}
    "$NVCC" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v "$HERE/$bin.cu" -o "$BUILD_DIR/$bin" \
      > "$OUT/build_$bin.log" 2>&1
    rc=$?; say "build $bin rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build_$bin.log"; exit 3; }
    "${CUDA_HOME:-/usr/local/cuda-13.0}/bin/cuobjdump" -sass "$BUILD_DIR/$bin" > "$OUT/$bin.sass"
    python3 "$HERE/sass_loops.py" "$OUT/$bin.sass" "$HERE/pins.json" > "$OUT/gate_$bin.jsonl"
    rc=$?; say "SASS gate $bin rc $rc"; [ "$rc" = 0 ] || { grep '"ok": false' "$OUT/gate_$bin.jsonl"; exit 4; }
  done
  if has dgemm; then
    lib=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)
    "$NVCC" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -DDG_CUBLASLT -lcublasLt -Xlinker "-rpath=$lib" \
      benchmarks/pouw/dc/dgemm.cu -o "$BUILD_DIR/dgemm" > "$OUT/build_dgemm.log" 2>&1
    rc=$?; say "build dgemm rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build_dgemm.log"; exit 3; }
    "${CUDA_HOME:-/usr/local/cuda-13.0}/bin/cuobjdump" -sass "$BUILD_DIR/dgemm" > "$OUT/dgemm.sass"
    n=$(grep -c IGMMA "$OUT/dgemm.sass"); say "dgemm SASS IGMMA $n"; [ "$n" -gt 0 ] || exit 4
  fi
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-20}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 200 \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
worst=0
if has bench; then
  SIZES=${SIZES:-"131072 458752 524288 1048576 3670016 4194304 6291456 14680064 16777216 29360128 33554432 51380224 58720256
                  67108864 262668288 1073741824"}
  "$BUILD_DIR/sw_bench" check > "$OUT/bench_check.jsonl" 2>&1
  rc=$?; say "sw_bench check rc $rc"; cat "$OUT/bench_check.jsonl"
  if [ "$rc" = 0 ]; then
    # shellcheck disable=SC2086
    "$BUILD_DIR/sw_bench" run "${REPS:-7}" $SIZES > "$OUT/bench.jsonl" 2> "$OUT/bench.err"
    rc=$?; say "sw_bench run rc $rc ($(wc -l < "$OUT/bench.jsonl") lines)"
  fi
  [ "$rc" = 0 ] || worst=$rc
fi
if has k32; then
  "$BUILD_DIR/k32" check > "$OUT/k32_check.jsonl" 2>&1
  rc=$?; say "k32 check rc $rc"; cat "$OUT/k32_check.jsonl"
  if [ "$rc" = 0 ]; then
    "$BUILD_DIR/k32" run "${REPS:-7}" "${STEPS:-65536}" > "$OUT/k32.jsonl" 2> "$OUT/k32.err"
    rc=$?; say "k32 run rc $rc ($(wc -l < "$OUT/k32.jsonl") lines)"
  fi
  [ "$rc" = 0 ] || worst=$rc
fi
if has dgemm; then
  # dgemm gate: C and R poisoned, the fold kernel against the host, the no-write control rejected; dgemm_ref.py rechecks
  exact=1
  for g in "128 256 256 7" "256 512 1024 11" "192 384 2048 13"; do
    read -r m n k seed <<<"$g"
    d="$OUT/dgemm/gate-${m}x${n}x${k}"
    mkdir -p "$d"
    "$BUILD_DIR/dgemm" gate "$d" "$m" "$n" "$k" "$seed" > "$d/result.json" || exact=0
    python3 benchmarks/pouw/dc/dgemm_ref.py "$d" > "$d/ref.json" || exact=0
    rm -f "$d/X.bin" "$d/W.bin"
  done
  say "dgemm gates exact $exact"
  if [ "$exact" = 1 ]; then
    SHAPES=${DGEMM_SHAPES:-"512,3072,2048 512,2048,2048 512,16384,2048 512,2048,8192 8192,3072,2048 8192,2048,2048
                            8192,16384,2048 8192,2048,8192 8192,7168,7168 8192,4096,7168 8192,7168,2048 8192,8192,8192"}
    for s in $SHAPES; do
      # shellcheck disable=SC2086
      "$BUILD_DIR/dgemm" bench "${REPS:-7}" ${s//,/ } >> "$OUT/dgemm.jsonl" 2>> "$OUT/dgemm.err" || worst=1
      say "dgemm bench $s: $(tail -1 "$OUT/dgemm.jsonl" | cut -c1-200)"
    done
  else
    worst=4
  fi
fi
exit "$worst"
