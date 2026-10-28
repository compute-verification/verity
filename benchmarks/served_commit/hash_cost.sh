#!/usr/bin/env bash
# hash_cost.py on one H200 in one short gpu-lease: hm96-sha512 rows alone, the Merkle tree, and the leaves beside the served int8 GEMM
# (Llama-3.2-1B at T 8,192) at a range of hash shares, the SM clock sampled beside it all. The libraries are built before the lease.
# Measurement only. cwd: the source tree's root.
#   VERITY_BENCH_VENV (the node's torch + triton environment), CUDA_HOME (/usr/local/cuda-13.0), LEASE_MIN (20), and hash_cost.py's
#   ROWS, TOTAL_GB, THREADS, WARPS, T, HASH_MS, REPS, STEP, GAP_CYCLES.
set -uo pipefail
. benchmarks/harness/env.sh
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/hash-cost-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) python $VENV/bin/python into $BUILD_DIR"
  "$VENV/bin/python" benchmarks/served_commit/hash_cost.py build > "$OUT/build.log" 2>&1
  rc=$?; say "build rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  if [[ "+${ARMSET:-}+" == *"+wgmma+"* ]]; then
    "$VENV/bin/python" -c "from verity_vllm.engine import int8_wgmma; print(int8_wgmma.lib())" >> "$OUT/build.log" 2>&1
    rc=$?; say "int8_wgmma build rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  fi
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-20}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

# ARMSET: one set or several joined by + (best+paired), run in order in one lease
has() { [[ "+${ARMSET:-}+" == *"+$1+"* ]]; }
if has diag; then
  D=THREADS=256,TREE=0
  ARMS="${ARMS:-} base:$D,ROWS=4096+65536,CAPS=1+2+4+8+16 base16k:$D,ROWS=16384,CAPS=2+4+8
        stride0:$D,ROWS=4096+65536,CAPS=2+8+16+132+0,STRIDE0=1 fused:$D,ROWS=4096+65536,CAPS=2+8+16+132+0,VERITY_ROWS_FORM=fused
        ph16:$D,ROWS=4096+65536,CAPS=2+8+16+132,VERITY_ROWS_PHASES=16 gdef:$D,ROWS=4096+65536,CAPS=8+16+0,GPRIO=0"
fi
# ARMSET=lifetime: the GEMM's loss against the compressions a hash launch runs a thread (VERITY_ROWS_PHASES), at T 8,192 and 512
if has lifetime; then
  D=THREADS=256,TREE=0
  ARMS="${ARMS:-} p8_4k:$D,ROWS=4096,VERITY_ROWS_PHASES=8,CAPS=16+33+66+132+264 p33_4k:$D,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=16+33+66+132+264+0
        p16_16k:$D,ROWS=16384,VERITY_ROWS_PHASES=16,CAPS=8+16+33+66+132 p64_16k:$D,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=16+33+66+132+264
        p64_64k:$D,ROWS=65536,VERITY_ROWS_PHASES=64,CAPS=8+16+33+66+132
        p33_4k_t512:$D,T=512,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=2+8+16+33+66+132
        p64_16k_t512:$D,T=512,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=2+8+16+33+66+132"
fi
# ARMSET=best: the cheapest phases found (about 1-2 compressions a launch a thread at 4 and 16 KiB, 8 at 64 KiB, the clamp), finer caps
if has best; then
  D=THREADS=256,TREE=0
  ARMS="${ARMS:-} b4k:$D,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=33+66+99+132+198+264+396+0
        b16k:$D,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=33+66+99+132+198+264+396+0
        b64k:$D,ROWS=65536,VERITY_ROWS_PHASES=64,CAPS=8+16+24+33+66+132+264+0
        b4k_t512:$D,T=512,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=1+2+4+8+16+33+66
        b16k_t512:$D,T=512,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=1+2+4+8+16+33+66
        b64k_t512:$D,T=512,ROWS=65536,VERITY_ROWS_PHASES=64,CAPS=1+2+4+8+16+33+66"
fi
# ARMSET=paired: the low-share points again with 4x the hash bytes, to bound what "free" costs a byte
if has paired; then
  D=THREADS=256,TREE=0,HASH_MS=120
  ARMS="${ARMS:-} a4k:$D,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=16+33+66+132+264
        a16k:$D,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=16+33+66+132
        a64k:$D,ROWS=65536,VERITY_ROWS_PHASES=64,CAPS=8+16+33+66
        a16k_p4:$D,ROWS=16384,CAPS=2+16+132
        a4k_t512:$D,T=512,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=2+8+16+33
        a16k_t512:$D,T=512,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=2+8+16+33"
fi
# ARMSET=step: beside a loop shaped like the served step (STEP=1: the non-GEMM kernels between the linears, the host gap after each
# forward) at the cheapest phases and the served 4, then GEMM-only controls (STEP=0) with the clock sampled in each leg
if has step; then
  D=THREADS=256,TREE=0,HASH_MS=60
  ARMS="${ARMS:-} s4k:$D,STEP=1,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=33+132+396+0
        s16k:$D,STEP=1,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=33+132+396+0
        s64k:$D,STEP=1,ROWS=65536,VERITY_ROWS_PHASES=64,CAPS=16+66+264+0
        s16k_p4:$D,STEP=1,ROWS=16384,CAPS=132+0
        s16k_nogap:$D,STEP=1,GAP_CYCLES=0,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=132+0
        g4k:$D,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=33+132+396+0
        g16k:$D,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=33+132+396+0"
fi
# ARMSET=wgmma: the same beside int8_fused's persistent wgmma GEMM (GEMM=wgmma; a source tree with engine/int8_wgmma.py), bare and step
if has wgmma; then
  D=THREADS=256,TREE=0,HASH_MS=60,GEMM=wgmma
  ARMS="${ARMS:-} w4k:$D,ROWS=4096,VERITY_ROWS_PHASES=33,CAPS=33+132+396+0
        w16k:$D,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=33+132+396+0
        w64k:$D,ROWS=65536,VERITY_ROWS_PHASES=64,CAPS=16+66+264+0
        ws16k:$D,STEP=1,ROWS=16384,VERITY_ROWS_PHASES=64,CAPS=33+132+396+0
        ws16k_p4:$D,STEP=1,ROWS=16384,CAPS=132+0"
fi
say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 200 \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
if [ -z "${ARMS:-}" ]; then
  "$VENV/bin/python" benchmarks/served_commit/hash_cost.py 2>&1 | tee "$OUT/hash_cost.log"
  rc=${PIPESTATUS[0]}
  say "hash_cost rc $rc"
  exit "$rc"
fi
# ARMS: space-separated NAME:K=V,K=V,... specs (a list's items joined by +), each one hash_cost.py process with that environment, its
# output in $OUT/NAME. ARMSET=diag: low caps, then what the GEMM's loss comes from (row traffic, the salt kernel's shared memory, how
# long a CTA lives, the GEMM's priority).
worst=0
for arm in $ARMS; do
  name=${arm%%:*}; kv=${arm#*:}; [ "$kv" = "$arm" ] && kv=
  mkdir -p "$OUT/$name"
  # shellcheck disable=SC2086
  env OUT="$OUT/$name" ${kv//,/ } "$VENV/bin/python" benchmarks/served_commit/hash_cost.py > "$OUT/$name/hash_cost.log" 2>&1
  rc=$?; say "arm $name ($kv) rc $rc"; grep -E "cap|failed" "$OUT/$name/hash_cost.log" | cut -c1-200
  [ "$rc" = 0 ] || worst=$rc
done
exit "$worst"
