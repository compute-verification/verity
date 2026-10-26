#!/usr/bin/env bash
# The bandwidth audit's receipts on a GPU against serve.py's fixture and the CPU reference, and their cost at the store's
# size (bw_check.py), from build.sh's lib for the toy's decode path; cwd = the source tree's root:
#
#   research run --on vy-dc-h200-3 --project verity --campaign e2e-demo --source . --cwd source --timeout 30m \
#     --env GPU_LEASE_WHO=kernels-kstage --send /tmp/p2lib.tar -- bash benchmarks/torch_gpu/p2_stage/bw.sh
#
# Under one untimed lease of at most LEASE_MIN minutes (default 10), refused near INF-242's window (Tue 27 Oct
# 08:00-10:00Z). RANDOM units of random C* (default 262144) against the reference, SAMPLE of them (default 4000) from
# their blocks. Each step's exit code is in $OUT/rc.txt; exits non-zero if any failed.
set -uo pipefail
O=${OUT:?}; IN=${RESEARCH_RUN_DIR:?}/inputs
D=benchmarks/torch_gpu/p2_stage
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
CUDA=${CUDA:-/usr/local/cuda-13.0}
RANDOM_UNITS=${RANDOM_UNITS:-262144}
SAMPLE=${SAMPLE:-4000}
LEASE_MIN=${LEASE_MIN:-10}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
export O D PY LEASE_MIN RANDOM_UNITS SAMPLE PATH=$CUDA/bin:$PATH P2_RECEIPTS_BUILD=$S/p2r SHA512_ROWS_BUILD=$S/sxr \
  PYTHONPATH=$PWD:$PWD/benchmarks/pous${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }

mkdir -p "$S/lib" && tar -C "$S/lib" -xf "$IN/p2lib.tar" && (cd "$S/lib" && sha256sum -c --quiet MANIFEST.sha256)
step ship $?
export LIB=$S/lib/lib
git rev-parse HEAD > "$O/commit.txt" 2>&1
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/python.txt" 2>&1
nvcc --version > "$O/nvcc.txt" 2>&1
gpu-lease status > "$O/lease-before.txt" 2>&1
echo "bw: asking for the GPU at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "$LEASE_MIN" -- bash -c '
  step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }
  echo "leased at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"
  now=$(date -u +%s); w0=$(date -u -d 2026-10-27T08:00:00Z +%s); w1=$(date -u -d 2026-10-27T10:00:00Z +%s)
  if [ $((now + 60 * LEASE_MIN)) -gt "$w0" ] && [ "$now" -lt "$w1" ]; then
    echo "a lease from now could reach INF-242 window (Tue 27 Oct 08:00-10:00Z): released unused" >> "$O/rc.txt"
    step window 3; exit 3
  fi
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,uuid,name,driver_version,clocks.max.sm,memory.total,memory.used \
    --format=csv > "$O/gpus.txt" 2>&1
  "$PY" "$D/bw_check.py" --out "$O" --lib "$LIB" --random "$RANDOM_UNITS" --sample "$SAMPLE" --bench \
    > "$O/bw_check.log" 2> "$O/bw_check.err"
  step bw_check $?
  echo "released at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"'
step lease $?
tail -1 "$O/bw_check.log"
cat "$O/rc.txt"
! grep -qE " rc [1-9]" "$O/rc.txt"
