#!/bin/bash
# The recorded measurement: builds every kernel into a scratch directory, then (on one leased GPU) the device probe, the
# exchange probe and the squaring chains with their pow checks and root identities, all into $RESEARCH_RUN_DIR/out.
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign pous --declared-output 'out/*' -- \
#     gpu-lease 1 --wait --on 0,1,2,3 --max-min 30 -- taskset -c 0-7 bash benchmarks/pous/p2_gpu_root/run.sh [N] [CONFIGS]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$out"
b=$(mktemp -d)
trap 'rm -rf "$b"' EXIT
nv=${NVCC:-/usr/local/cuda/bin/nvcc}
f="-O3 -arch=sm_120 -std=c++17"
$nv $f -o "$b/probe" "$here/probe.cu"
$nv $f -o "$b/xchg" "$here/xchg.cu"
$nv $f -DLB=512 -o "$b/sqchain" "$here/sqchain.cu"
$nv $f -DLB=1024 -o "$b/sqchain_lb1024" "$here/sqchain.cu"
$nv $f -DR=4 -o "$b/sqchain2_r4" "$here/sqchain2.cu"
$nv $f -DR=8 -o "$b/sqchain2_r8" "$here/sqchain2.cu"
$nv $f -o "$b/probe2" "$here/probe2.cu"
$nv $f -o "$b/sqchain3" "$here/sqchain3.cu"
n=${1:-20000}
configs=${2:-sqchain_lb1024:1:1024:0,sqchain:4:512:0,sqchain:8:256:0,sqchain:8:256:1,sqchain:16:256:1,sqchain2_r4:1:1024:1,sqchain2_r4:8:704:1,sqchain2_r4:16:352:1,sqchain2_r8:16:192:1,sqchain3:12:256:4,sqchain3:20:512:4,sqchain3:28:512:4,sqchain3:8:256:4,sqchain3:12:512:4,sqchain3:60:1024:4,sqchain3:633:1024:1}
nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-before.txt"
"$b/probe" > "$out/probe.jsonl"
"$b/probe2" > "$out/probe2.jsonl"
"$b/xchg" 256 > "$out/xchg.jsonl"
python3 "$here/driver.py" --out "$out" --n "$n" --reps 5 --roots 2 --configs "$configs" \
  --bin sqchain="$b/sqchain" --bin sqchain_lb1024="$b/sqchain_lb1024" --bin sqchain2_r4="$b/sqchain2_r4" --bin sqchain2_r8="$b/sqchain2_r8" --bin sqchain3="$b/sqchain3"
nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-after.txt"
