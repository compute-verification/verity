#!/usr/bin/env bash
# sis_bench on one leased GPU (cwd = the source tree's root). Outside the lease: builds sis_bench and commit_gpu (from its own
# path, read-only) for this GPU's architecture. Under one lease: commit_gpu's raw SHA-512 rate, every CONFIG through
# sis_bench (its bit-exactness gate first, then interleaved reps), commit_gpu's rate again. After the lease: reference.py
# on every dump. Clocks are read, never set.
#
#   research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source \
#     --env GPU_LEASE_WHO=hash-lattice [--env LEASE="--preemptible"] -- bash benchmarks/hash/lattice/gpu.sh
# CONFIGS: ENC:M_BYTES:N_WORDS[:tree] items; BATCH_MIB (default 1024); REPS (default 10); LEASE_MIN (default 14).
set -uo pipefail
O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
D=benchmarks/hash/lattice
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $cc in 90|100|120) arch=${cc}a ;; *) arch=$cc ;; esac
echo "arch $arch" > "$O/rc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_$arch,code=sm_$arch -Xptxas -v -o "$W/sis_bench" $D/sis_bench.cu \
  -lcublasLt -lnvidia-ml > "$O/build-sis.log" 2>&1
echo "build sis rc $?" >> "$O/rc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_$arch,code=sm_$arch -DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1 \
  -o "$W/commit_gpu" benchmarks/pouw/commit_gpu/commit_gpu.cu -lcublasLt -lnvidia-ml > "$O/build-commit-gpu.log" 2>&1
echo "build commit_gpu rc $?" >> "$O/rc.txt"
mkdir -p "$O/dumps"
export O W
export CONFIGS=${CONFIGS:-"q32:4096:256:tree q32:16384:256:tree q16:4096:512:tree q16:16384:512:tree q8n:4096:384 q8n:16384:384 q32:16384:176 q32:16384:208 q16:16384:352 q16:16384:416 q8n:16384:272 q8n:16384:320"}
export BATCH_MIB=${BATCH_MIB:-1024} REPS=${REPS:-10}
# shellcheck disable=SC2086,SC2016
gpu-lease 1 --wait ${LEASE:-} --max-min "${LEASE_MIN:-14}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.total --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" \
    > "$O/smi.csv" 2>&1 &
  smi=$!
  [ -x "$W/commit_gpu" ] && "$W/commit_gpu" sha 20 > "$O/commit-gpu-sha-before.jsonl" 2>> "$O/bench.err"
  for c in $CONFIGS; do
    IFS=: read -r enc m n tree <<< "$c"
    "$W/sis_bench" "$enc" "$m" "$n" "$BATCH_MIB" "$REPS" "$O/dumps" $tree >> "$O/bench.jsonl" 2>> "$O/bench.err"
    echo "sis $c rc $?" >> "$O/rc.txt"
  done
  [ -x "$W/commit_gpu" ] && "$W/commit_gpu" sha 20 > "$O/commit-gpu-sha-after.jsonl" 2>> "$O/bench.err"
  kill $smi
'
echo "lease rc $?" >> "$O/rc.txt"
python3 $D/reference.py "$O/dumps" > "$O/reference.jsonl" 2>&1
echo "reference rc $?" >> "$O/rc.txt"
python3 $D/summary.py "$O" > "$O/summary.md" 2>&1
cat "$O/rc.txt"
