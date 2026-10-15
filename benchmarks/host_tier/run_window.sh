#!/usr/bin/env bash
# The whole-node window on vy-nebius-2 (every CPU and all 8 GPUs, the node kept quiet) as one Attempt, started inside the
# window (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- env UNTIL=EPOCH taskset -c 0-191 bash benchmarks/host_tier/run_window.sh
#
# Builds once, then under one `gpu-lease 8 --wait --timed` runs in turn, each step's outputs under out/<step>/out:
# run_loop.sh (SWEEP=window) on the whole node with 8 GPUs, on socket 0 with GPUs 0-3 and on socket 1 with GPUs 4-7;
# run_pcie.sh with all 8; run_dram.sh on socket 0, socket 1 and both. A step whose estimate would end after UNTIL is skipped,
# and every step is stopped at UNTIL. Socket 0 is CPUs 0-95, socket 1 96-191 (siblings 2k, 2k+1): each socket's first two
# cores drive its GPUs, its third holds the probe, and the other 45 are workers.
# Env: UNTIL (epoch seconds, required), REPS (6), SKIP (step names, space-separated).
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, steps.jsonl, numa.txt, and each step's out/.
set -uo pipefail
cd "$(dirname "$0")"
R=${RESEARCH_RUN_DIR:-$PWD}; mkdir -p "$R/out"
: "${UNTIL:?epoch seconds the window ends}"

if [ "${1:-}" = leased ]; then
  B=$2 RP=${REPS:-6}
  uuid() { nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader | paste -sd,; }
  step() {  # step NAME MINUTES ENV...: run ENV's command with its outputs under out/NAME, unless it can't end by UNTIL
    local name=$1 min=$2 t0; shift 2
    t0=$(date +%s)
    if [ $((t0 + min * 60)) -gt "$UNTIL" ] || [[ " ${SKIP:-} " == *" $name "* ]]; then
      echo "{\"step\": \"$name\", \"skipped\": true, \"at\": $t0}" | tee -a "$R/out/steps.jsonl"
      return
    fi
    numactl -H > "$R/out/numa-$name.txt" 2>&1
    timeout $((UNTIL - t0)) env RESEARCH_RUN_DIR="$R/out/$name" HOST_TIER_BIN="$B" HOST_TIER_LEASED=1 "$@"
    echo "{\"step\": \"$name\", \"rc\": $?, \"start\": $t0, \"end\": $(date +%s)}" | tee -a "$R/out/steps.jsonl"
  }
  all=0,1,2,3,4,5,6,7 s0=0,1,2,3 s1=4,5,6,7
  step loop-node 10 CUDA_VISIBLE_DEVICES=$all GPU_LEASE_UUID="$(uuid $all)" NGPU=8 SWEEP=window REPS="$RP" GPU_NODE=local \
    WORKERS=6-95,102-191 HOST_CPUS=0,1,2,3,96,97,98,99 PROBE_CPU=4 PROBE_REPS=1 bash run_loop.sh
  step loop-s0 7 CUDA_VISIBLE_DEVICES=$s0 GPU_LEASE_UUID="$(uuid $s0)" NGPU=4 SWEEP=window REPS="$RP" GPU_NODE=0 \
    WORKERS=6-95 HOST_CPUS=0,1,2,3 PROBE_CPU=4 PROBE_REPS=0 bash run_loop.sh
  step loop-s1 7 CUDA_VISIBLE_DEVICES=$s1 GPU_LEASE_UUID="$(uuid $s1)" NGPU=4 SWEEP=window REPS="$RP" GPU_NODE=1 \
    WORKERS=102-191 HOST_CPUS=96,97,98,99 PROBE_CPU=100 PROBE_REPS=0 bash run_loop.sh
  step pcie 5 CUDA_VISIBLE_DEVICES=$all GPU_LEASE_UUID="$(uuid $all)" NGPU=8 REPS=2 SECONDS_EACH=2 GRIDS="188 752 3008" \
    PCIE_CPUS=0-3,96-99 DRAM_CPUS=6-95,102-191 HOST_NODE=local bash run_pcie.sh
  step dram-s0 3 CPUS=0-95 NODE=0 REMOTE=1 REPS=1 SECONDS_EACH=2 bash run_dram.sh
  step dram-s1 3 CPUS=96-191 NODE=1 REMOTE=0 REPS=1 SECONDS_EACH=2 bash run_dram.sh
  step dram-node 4 CPUS=0-191 NODE=local REMOTE=i REPS=1 SECONDS_EACH=2 bash run_dram.sh
  exit 0
fi

B=$(mktemp -d /tmp/host-tier-window.XXXXXX); trap 'rm -rf "$B"' EXIT
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
{ "$NVCC" --version && g++ -O3 -march=native -Wall -c receipt.cpp -o "$B/receipt.o" \
    && "$NVCC" -O3 -std=c++17 -arch=sm_120 -Xptxas -v -Xcompiler -pthread -o "$B/loop" loop.cu "$B/receipt.o" \
    && "$NVCC" -O3 -std=c++17 -arch=sm_120 -o "$B/pcie" pcie.cu && gcc -O3 -march=native -pthread -Wall -o "$B/dram" dram.c; } \
  > "$R/out/build.txt" 2>&1 || { cat "$R/out/build.txt"; exit 4; }
{ taskset -pc $$; cat /sys/fs/cgroup/user.slice/cpuset.cpus.effective; numactl -H; } > "$R/out/numa.txt" 2>&1
M=$(( (UNTIL - $(date +%s)) / 60 ))
[ "$M" -ge 5 ] || { echo "window over"; exit 5; }
GPU_LEASE_WHO=ra-host-tier gpu-lease 8 --wait --timed --max-min "$M" -- bash "$PWD/run_window.sh" leased "$B"
