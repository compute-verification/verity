#!/bin/bash
# The recorded measurement: builds every kernel into a scratch directory, then (on one leased GPU) the device probe, the
# exchange probe and the squaring chains with their pow checks and root identities, all into $RESEARCH_RUN_DIR/out.
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign pous --declared-output 'out/*' -- \
#     gpu-lease 1 --wait --on 0,1,2,3 --max-min 30 -- taskset -c 0-7 bash benchmarks/pous/p2_gpu_root/run.sh [N] [CONFIGS] [SET]
# SET is all (default) or rns: only the RNS chains (rnschain, rns.py) and the floor model.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$out"
b=$(mktemp -d)
trap 'rm -rf "$b"' EXIT
nv=${NVCC:-/usr/local/cuda/bin/nvcc}
f="-O3 -arch=sm_120 -std=c++17"
n=${1:-20000}
set_=${3:-all}
$nv $f -DRR=7 -DNC4=6 -DJJ=1 -DTPR=64 -o "$b/rns_r7_j1" "$here/rnschain.cu"
$nv $f -DRR=10 -DNC4=5 -DJJ=1 -DTPR=64 -o "$b/rns_r10_j1" "$here/rnschain.cu"
$nv $f -DRR=14 -DNC4=11 -DJJ=1 -DTPR=32 -o "$b/rns_r14_j1" "$here/rnschain.cu"
$nv $f -DRR=14 -DNC4=21 -DJJ=2 -DTPR=32 -o "$b/rns_r14_j2" "$here/rnschain.cu"
rns() {
  for h in 447 324 0; do python3 "$here/floor_model.py" --hop "$h" --json > "$out/floor_model_hop$h.json"; done
  python3 "$here/rns.py" --out "$out" --n "$n" --reps 5 --roots 2 \
    --configs r7j1:188:7:1,r10j1:128:10:1,r14j1:94:14:1,r14j2:188:14:2 \
    --bin r7j1="$b/rns_r7_j1" --bin r10j1="$b/rns_r10_j1" --bin r14j1="$b/rns_r14_j1" --bin r14j2="$b/rns_r14_j2"
}
if [ "$set_" = rns ]; then
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-before.txt"
  rns
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-after.txt"
  exit 0
fi
$nv $f -o "$b/probe" "$here/probe.cu"
$nv $f -o "$b/xchg" "$here/xchg.cu"
$nv $f -DLB=512 -o "$b/sqchain" "$here/sqchain.cu"
$nv $f -DLB=1024 -o "$b/sqchain_lb1024" "$here/sqchain.cu"
$nv $f -DR=4 -o "$b/sqchain2_r4" "$here/sqchain2.cu"
$nv $f -DR=8 -o "$b/sqchain2_r8" "$here/sqchain2.cu"
$nv $f -o "$b/probe2" "$here/probe2.cu"
$nv $f -o "$b/sqchain3" "$here/sqchain3.cu"
$nv $f -o "$b/sqchain4" "$here/sqchain4.cu"
$nv $f -DREPL=1 -o "$b/sqchain4_r1" "$here/sqchain4.cu"
$nv $f -o "$b/probe3" "$here/probe3.cu"
$nv $f -o "$b/probe4" "$here/probe4.cu"
configs=${2:-sqchain_lb1024:1:1024:0,sqchain:4:512:0,sqchain:8:256:0,sqchain:8:256:1,sqchain:16:256:1,sqchain2_r4:1:1024:1,sqchain2_r4:8:704:1,sqchain2_r4:16:352:1,sqchain2_r8:16:192:1,sqchain3:12:256:4,sqchain3:20:512:4,sqchain3:28:512:4,sqchain3:8:256:4,sqchain3:12:512:4,sqchain3:60:1024:4,sqchain3:636:1024:1,sqchain4:8:192,sqchain4:12:256,sqchain4:4:128,sqchain4:16:320,sqchain4:20:384,sqchain4:28:512,sqchain4_r1:8:192,sqchain4_r1:12:256}
nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-before.txt"
"$b/probe" > "$out/probe.jsonl"
"$b/probe2" > "$out/probe2.jsonl"
"$b/probe3" > "$out/probe3.jsonl"
"$b/probe4" > "$out/probe4.jsonl"
"$b/xchg" 256 > "$out/xchg.jsonl"
python3 "$here/driver.py" --out "$out" --n "$n" --reps 5 --roots 2 --configs "$configs" \
  --bin sqchain="$b/sqchain" --bin sqchain_lb1024="$b/sqchain_lb1024" --bin sqchain2_r4="$b/sqchain2_r4" --bin sqchain2_r8="$b/sqchain2_r8" --bin sqchain3="$b/sqchain3" --bin sqchain4="$b/sqchain4" --bin sqchain4_r1="$b/sqchain4_r1"
rns
nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-after.txt"
