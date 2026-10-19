#!/usr/bin/env bash
# The pipe probe on one leased GPU (cwd = the source tree's root): whether SHA-512's integer work hides under tensor-core
# work on one SM, apart from any GEMM's quality (fuse.cu's probe_kernel; on sm_90 also wgmma), and the static SASS opcode
# mix of fuse's GEMM, fused-kernel and separate-leaf kernels. Timing only. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-structure \
#     -- bash benchmarks/hash/structure/probe.sh
# REPS (default 9); LEASE_MIN (default 10); LEASE_ARGS (extra gpu-lease flags)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:/usr/local/cuda-13.0/bin:$PATH
S=benchmarks/hash/structure CG=benchmarks/pouw/commit_gpu
W=$(mktemp -d /dev/shm/hash-probe.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case $cap in 120) arch=compute_120a,code=sm_120a ;; 90) arch=compute_90a,code=sm_90a ;; *) arch=compute_${cap},code=sm_${cap} ;; esac
nvcc -O3 -std=c++17 -gencode arch=$arch -Xptxas -v -I$CG -o "$W/fuse" $S/fuse.cu -lcublasLt -lnvidia-ml > "$O/build-fuse.log" 2>&1
[ -x "$W/fuse" ] || { echo "build fuse FAILED"; tail -30 "$O/build-fuse.log"; exit 1; }
grep -iE "registers|spill|wgmma|serializ" "$O/build-fuse.log" | sed 's/^ptxas info *: //' > "$O/regs-fuse.txt"
cuobjdump -sass "$W/fuse" | awk '
  /Function : / { fn = $3; next }
  /\/\*[0-9a-f]+\*\/ / && fn ~ /gemm_kernel|sep_leaves|probe_kernel/ {
    sub(/.*\/\*[0-9a-f]+\*\/ +/, ""); split($0, t, /[ ;]+/); op = t[1]; if (op ~ /^@/) op = t[2]
    sub(/\..*/, "", op); n[fn, op]++; tot[fn]++
  }
  END { for (k in n) { split(k, p, SUBSEP); printf "%s\t%s\t%d\t%d\n", p[1], p[2], n[k], tot[p[1]] } }' \
  | sort -k1,1 -k3,3nr > "$O/sass-mix.tsv"
export REPS=${REPS:-9}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-10}" ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$W/fuse" probe "$REPS" > "$O/probe.jsonl" 2> "$O/probe.err"
  echo "probe rc $?"'
cat "$O/probe.jsonl"
