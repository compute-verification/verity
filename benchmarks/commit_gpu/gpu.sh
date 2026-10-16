#!/usr/bin/env bash
# commit_gpu on one leased sm_120 GPU (cwd = the source tree's root). Outside the lease: builds (the pair-add and native
# 64-bit SHA-512 forms, and a timing-only build without M_key y) and reference.py's cases on the CPU. Under one lease: the
# selftest, both builds' bit-exactness check against the reference (a build is timed only if its check passed), the raw
# compression rate, the e2e slice's o_proj operand A (8192 x 4096 int8, 2^20 leaves) committed alone and beside cuBLASLt's
# int8 o_proj GEMM, and the leaf-size sweep over the same operand as raw int8 rows. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=COMMIT-GPU \
#     -- bash benchmarks/commit_gpu/gpu.sh
# CHECK=quick|full (default full); REPS (default 9); BUILDS (tags a<SHA_ADD>r<SHA_ROT>[nomask]); SWEEP (int8 values a leaf).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/commit_gpu PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/commit-gpu.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nproc > "$O/nproc.txt"

build() {   # build TAG: a<SHA_ADD>r<SHA_ROT>[nomask]
  local t=$1 add=${1:1:1} rot flags=""
  rot=${t#a?r}; rot=${rot%nomask}
  case $t in *nomask) flags=-DCOMMIT_GPU_DIAG_NOMASK ;; esac
  nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -DSHA_ADD="$add" -DSHA_ROT="$rot" $flags \
    -o "$W/commit_gpu_$t" $D/commit_gpu.cu -lcublasLt -lnvidia-ml > "$O/build-$t.log" 2>&1
  echo "build $t rc $?" >> "$O/rc.txt"
}
export BUILDS=${BUILDS:-a0r0 a0r3 a0r7 a0r15 a1r3 a0r3nomask}
for t in $BUILDS; do build "$t" & done
$PY $D/reference.py prepare "$W/cases" "${CHECK:-full}" > "$O/prepare.jsonl" 2> "$O/prepare.err"
echo "prepare rc $?" >> "$O/rc.txt"
export SWEEP=${SWEEP:-32 64 128 256 512 1024 2048 4096}
$PY $D/reference.py params "$W/p-e2e.bin" 1048576 1024
for L in $SWEEP; do $PY $D/reference.py params "$W/p-rows-$L.bin" $((33554432 / L)) $((8 * L)); done
wait
cat "$O/rc.txt"

export REPS=${REPS:-9}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 20 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" \
    > "$O/gpu.txt" 2>&1
  timed=""
  for t in $BUILDS; do
    [ -x "$W/commit_gpu_$t" ] || continue
    case $t in
      *nomask) timed="$timed $t" ;;
      *) "$W/commit_gpu_$t" selftest >> "$O/selftest.jsonl"; s=$?
         $PY $D/reference.py check "$W/commit_gpu_$t" "$W/cases" "$t" > "$O/check-$t.jsonl" 2> "$O/check-$t.err"; c=$?
         echo "selftest $t rc $s; check $t rc $c" >> "$O/rc.txt"
         [ $s = 0 ] && [ $c = 0 ] && timed="$timed $t" ;;
    esac
  done
  echo "timed:$timed" >> "$O/rc.txt"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for t in $timed; do "$W/commit_gpu_$t" sha 20 > "$O/sha-$t.jsonl" 2>> "$O/bench.err"; done
  for t in $timed; do
    "$W/commit_gpu_$t" bench "$W/p-e2e.bin" e2e 1048576 1024 "$REPS" 8192 4096 4096 > "$O/bench-e2e-$t.jsonl" 2>> "$O/bench.err"
    echo "bench e2e $t rc $?" >> "$O/rc.txt"
  done
  best=$($PY $D/summary.py best "$O")
  echo "best: $best" >> "$O/rc.txt"
  for L in $SWEEP; do
    [ -n "$best" ] || break
    "$W/commit_gpu_$best" bench "$W/p-rows-$L.bin" rows $((33554432 / L)) $((8 * L)) "$REPS" 8192 4096 4096 \
      >> "$O/bench-sweep-$best.jsonl" 2>> "$O/bench.err"
    echo "bench rows L=$L $best rc $?" >> "$O/rc.txt"
  done
  kill $smi
  # the kernels'"'"' launch shapes (registers a thread, shared memory, blocks): can a commit kernel sit beside a GEMM CTA?
  N=$(command -v nsys || ls /usr/local/cuda-13.0/bin/nsys 2>/dev/null)
  if [ -n "$best" ] && [ -n "$N" ]; then
    "$N" profile -o "$W/trace" -t cuda --force-overwrite true "$W/commit_gpu_$best" bench "$W/p-rows-4096.bin" rows 8192 32768 2 \
      8192 4096 4096 > /dev/null 2> "$O/nsys.err"
    "$N" stats -r cuda_gpu_trace -f csv -o "$O/nsys" "$W/trace.nsys-rep" >> "$O/nsys.err" 2>&1
    echo "nsys rc $?" >> "$O/rc.txt"
  fi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
for f in "$O"/check-*.jsonl; do [ -e "$f" ] && tail -1 "$f"; done
$PY $D/summary.py "$O"
