#!/usr/bin/env bash
# The fused kernel taken apart on one leased GPU (cwd = the source tree's root): fuse check (every arm, the prefetching
# ones included, equal to the separate pass, and reference.py verify on the dumps), then `fuse diag` at Qwen3-8B's prefill
# GEMMs. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-structure \
#     -- bash benchmarks/hash/structure/diag.sh
# REPS (default 11); LEASE_MIN (default 10); LEASE_ARGS (extra gpu-lease flags)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:/usr/local/cuda-13.0/bin:$PATH
export S=benchmarks/hash/structure CG=benchmarks/pouw/commit_gpu PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/hash-diag.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case $cap in 120) arch=compute_120a,code=sm_120a ;; 90) arch=compute_90a,code=sm_90a ;; *) arch=compute_${cap},code=sm_${cap} ;; esac
nvcc -O3 -std=c++17 -gencode arch=$arch -Xptxas -v -I$CG -o "$W/fuse" $S/fuse.cu -lcublasLt -lnvidia-ml > "$O/build-fuse.log" 2>&1
[ -x "$W/fuse" ] || { echo "build fuse FAILED"; tail -30 "$O/build-fuse.log"; exit 1; }
for k in _Z11gemm_kernelILi0EEv3Geo _Z11gemm_kernelILi4EEv3Geo _Z11gemm_kernelILi8EEv3Geo; do
  cuobjdump -sass -fun "$k" "$W/fuse" > "$W/sass.txt" 2>/dev/null
  echo "$k IMMA $(grep -c IMMA "$W/sass.txt") NOP $(grep -c ' NOP ' "$W/sass.txt") lines $(wc -l < "$W/sass.txt")" >> "$O/sass-gemm.txt"
done
params() { $PY $CG/reference.py params "$W/p-$1-$2-$3.bin" $(( ($1 + $3 - 1) / $3 * (2 * $2 / 32) )) $((256 * $3)) --ks-seed 7; }
export CHECKS="200 384 128 64|4096 4096 4096 64|32 512 256 32" RUNS="4096 24576 4096 64|4096 4096 4096 64|4096 6144 4096 64|4096 4096 12288 64|4096 24576 4096 32"
IFS='|'; for c in $CHECKS $RUNS; do IFS=' '; set -- $c; [ -e "$W/p-$1-$2-$4.bin" ] || params "$1" "$2" "$4"; done; IFS=' '
export REPS=${REPS:-11}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-10}" ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  i=0; IFS="|"
  for c in $CHECKS; do
    IFS=" "; set -- $c; d="$W/dump$i"; mkdir -p "$d"
    "$W/fuse" check "$W/p-$1-$2-$4.bin" "$d" "$1" "$2" "$3" "$4" >> "$O/check-fuse.jsonl" 2>> "$O/check-fuse.err" || { echo "check $c FAILED"; exit 1; }
    i=$((i + 1)); IFS="|"
  done
  IFS="|"
  for c in $RUNS; do
    IFS=" "; set -- $c
    "$W/fuse" diag "$W/p-$1-$2-$4.bin" "$1" "$2" "$3" "$4" "$REPS" >> "$O/diag.jsonl" 2>> "$O/diag.err"; IFS="|"
  done'
echo "lease rc $?"
$PY $CG/reference.py verify "$W"/dump* > "$O/verify.jsonl" 2> "$O/verify.err"; echo "reference verify rc $?"
grep -c '"check": true' "$O/check-fuse.jsonl"
cat "$O/diag.jsonl"
