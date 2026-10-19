#!/usr/bin/env bash
# Each kernel's launch resources (grid, block, registers a thread, shared memory) in one `fuse diag` rep at gate_up's
# prefill shape, from an nsys trace on one leased GPU: whether a hash CTA can sit beside cuBLASLt's GEMM CTAs on an SM.
#
#   research run --on vy-nebius-2 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-structure \
#     -- bash benchmarks/hash/structure/kernels.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:/usr/local/cuda-13.0/bin:$PATH
S=benchmarks/hash/structure CG=benchmarks/pouw/commit_gpu
W=$(mktemp -d /dev/shm/hash-kernels.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case $cap in 120) arch=compute_120a,code=sm_120a ;; 90) arch=compute_90a,code=sm_90a ;; *) arch=compute_${cap},code=sm_${cap} ;; esac
nvcc -O3 -std=c++17 -gencode arch=$arch -I$CG -o "$W/fuse" $S/fuse.cu -lcublasLt -lnvidia-ml > "$O/build-fuse.log" 2>&1 || exit 1
python3 $CG/reference.py params "$W/p.bin" $(( 4096 / 64 * (2 * 24576 / 32) )) $((256 * 64)) --ks-seed 7
command -v nsys > "$O/nsys.txt" || { echo "no nsys"; exit 1; }
gpu-lease 1 --wait --max-min 8 ${LEASE_ARGS:-} -- bash -c 'nsys profile --trace=cuda --sample=none --cpuctxsw=none -o "$W/t" "$W/fuse" diag "$W/p.bin" 4096 24576 4096 64 1 > "$O/diag.jsonl" 2> "$O/nsys.err"'
nsys export --type sqlite -o "$W/t.sqlite" "$W/t.nsys-rep" >> "$O/nsys.err" 2>&1
python3 - "$W/t.sqlite" "$O/kernels.tsv" <<'EOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
q = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.registersPerThread,
       k.staticSharedMemory, k.dynamicSharedMemory, COUNT(*), AVG(k.end - k.start)
       FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
       GROUP BY s.value, k.gridX, k.blockX, k.registersPerThread, k.dynamicSharedMemory ORDER BY s.value"""
with open(sys.argv[2], "w") as f:
    f.write("kernel\tgrid\tblock\tregs\tsmem_static\tsmem_dynamic\tlaunches\tavg_us\n")
    for r in db.execute(q):
        f.write(f"{r[0][:80]}\t{r[1]}x{r[2]}x{r[3]}\t{r[4]}x{r[5]}x{r[6]}\t{r[7]}\t{r[8]}\t{r[9]}\t{r[10]}\t{r[11] / 1e3:.1f}\n")
EOF
cat "$O/kernels.tsv"
