#!/usr/bin/env bash
# N-ATTACK's probes on one H200 in one gpu-lease: the binary is built and its SASS gated (gate.py) before the lease; under
# it, the correctness check runs before the timing, the SM clock sampled beside it all. Measurement only.
# cwd: the source tree's root.
#   REPS (7), ITERS (20000, loop trips a thread), CUDA_HOME (/usr/local/cuda-13.0), LEASE_MIN (12)
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }
HERE=benchmarks/pouw/np_attack
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/np-attack-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) into $BUILD_DIR"
  "$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v "$HERE/nattack.cu" \
    -o "$BUILD_DIR/nattack" > "$OUT/build.log" 2>&1
  rc=$?; say "build rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  "$CUDA_HOME/bin/cuobjdump" -sass "$BUILD_DIR/nattack" > "$OUT/nattack.sass"
  python3 "$HERE/gate.py" "$OUT/nattack.sass" > "$OUT/gate.jsonl"
  rc=$?; say "SASS gate rc $rc"; cat "$OUT/gate.jsonl"
  [ "$rc" = 0 ] || { say "gate failed: no timing"; exit 4; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-12}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} GPU_LEASE_UUID=${GPU_LEASE_UUID-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 200 \
  -i "${GPU_LEASE_UUID:-${CUDA_VISIBLE_DEVICES:-0}}" > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
"$BUILD_DIR/nattack" check > "$OUT/check.jsonl" 2>&1
rc=$?; say "check rc $rc"; cat "$OUT/check.jsonl"
[ "$rc" = 0 ] || { say "check failed: no timing"; exit 5; }
"$BUILD_DIR/nattack" run "${REPS:-7}" "${ITERS:-20000}" > "$OUT/probes.jsonl" 2> "$OUT/probes.err"
rc=$?; say "run rc $rc ($(wc -l < "$OUT/probes.jsonl") lines)"; cat "$OUT/probes.jsonl"
exit "$rc"
