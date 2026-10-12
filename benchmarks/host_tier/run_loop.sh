#!/usr/bin/env bash
# The host tier's honest loop against the adversary's best read on the same CPUs and GPUs, and what the loop costs a
# vLLM-like host process, as one Attempt (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- env NGPU=1 taskset -c 48-95 bash benchmarks/host_tier/run_loop.sh
#
# Outside the lease: build loop.cu (+ receipt.cpp), pcie.cu and dram.c. Under `gpu-lease $NGPU --wait --max-min 20`: the
# loop's gates against reference.py (a failed gate times nothing, exit 3); then, REPS times, the GPUs' receipts alone over
# grids, the CPUs' receipts alone (one thread, one a core, every CPU; KS 4 and 16), the adversary's CPU read alone and
# beside the GPUs' receipts, the full loop (GPUs and CPUs), the loop on half the CPUs beside the adversary on the other
# half, and the adversary's CPU read beside each GPU's copy engines and zero-copy reads (dram.c and pcie.cu, as
# run_pcie.sh); then hostproc.py alone and beside the loop, the adversary, and the GPUs' receipts alone.
# All host-tier memory is on NUMA node GPU_NODE (0), the node of the worker CPUs: one socket's DRAM.
# Env: NGPU (1), REPS (2), SECONDS_EACH (3), LOOP_GRID (32), WORKERS (52-95), HOST_CPUS (GPU driver threads, 48,49),
#      PROBE_CPU (50), PROBE_SECONDS (5), PROBE_REPS (3), GPU_NODE (0).
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, gpu.txt, gate16/, gate4/, gates.jsonl, loop.jsonl, pcie.jsonl,
# dram.jsonl, hostproc.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"

if [ "${1:-}" = leased ]; then
  B=$2
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem \
    --format=csv > "$O/gpu.txt" 2>&1
  S=${SECONDS_EACH:-3} G=${LOOP_GRID:-32} W=${WORKERS:-52-95} H=${HOST_CPUS:-48,49} PR=${PROBE_CPU:-50} N=${NGPU:-1}
  L() { "$B/loop" --seconds "$S" --gpu-mib 2048 --cpu-mib 256 --adv-mib 256 --gpu-node "${GPU_NODE:-0}" --host-cpus "$H" "$@"; }
  one() {
    python3 cpuload.py snap > "$B/a.json"
    line=$(L "$@")
    load=$(python3 cpuload.py diff "$B/a.json" --exclude 48-95)
    echo "${line%\}}, \"load\": $load, \"cuda_visible\": \"$CUDA_VISIBLE_DEVICES\"}" | tee -a "$O/loop.jsonl"
  }
  for k in 16 4; do
    mkdir -p "$O/gate$k"
    L --gate "$O/gate$k" --cpus "${W%%[-,]*}" --ks "$k" || { echo "gate run failed"; exit 3; }
    python3 reference.py "$O/gate$k/gate.json" | tee -a "$O/gates.jsonl"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "gate failed"; exit 3; }
  done
  first=${W%%[-,]*}; last=${W##*-}; half=$(( (first + last + 1) / 2 ))
  for r in $(seq "${REPS:-2}"); do
    for g in 4 8 16 32 64 376; do one --label gpu --gpus "$N" --grid "$g"; done
    for k in 4 16; do for c in "$first" "$first-$((last - 1)):2" "$W"; do one --label cpu --gpus 0 --cpus "$c" --ks "$k"; done; done
    one --label adv --gpus 0 --adv-cpus "$W"
    one --label gpu+adv --gpus "$N" --grid "$G" --adv-cpus "$W"
    for k in 4 16; do one --label honest --gpus "$N" --grid "$G" --cpus "$W" --ks "$k"; done
    one --label honest-half+adv --gpus "$N" --grid "$G" --cpus "$first-$((half - 1))" --adv-cpus "$half-$last" --ks 4
    # the adversary's CPU read beside the GPUs' copy engines and zero-copy reads, timed together
    D() { "$B/dram" --cpus "$W" --node "${GPU_NODE:-0}" --kernel multi --thp 1 --mib 256 --seconds "$S" "$@"; }
    D --label cpu-alone | tee -a "$O/dram.jsonl"
    for m in h2d zcr; do
      t=$(python3 -c 'import time; print(time.time() + 8)')
      D --label "cpu-with-$m" --start-at "$t" > "$B/d.json" &
      taskset -c "$H" "$B/pcie" --seconds "$S" --mib 1024 --host-node "${GPU_NODE:-0}" --modes "$m" --start-at "$t" \
        | sed "s/^{/{\"label\": \"with-cpu-$m\", \"cuda_visible\": \"$CUDA_VISIBLE_DEVICES\", /" | tee -a "$O/pcie.jsonl"
      wait; tee -a "$O/dram.jsonl" < "$B/d.json"
    done
  done
  probe() { taskset -c "$PR" python3 hostproc.py --seconds "${PROBE_SECONDS:-5}" "$@" | tee -a "$O/hostproc.jsonl"; }
  for r in $(seq "${PROBE_REPS:-3}"); do
    probe --label alone
    for spec in "honest:--gpus $N --grid $G --cpus $W --ks 4" "adv:--gpus 0 --adv-cpus $W" "gpu:--gpus $N --grid $G"; do
      rm -f "$B/go"
      # shellcheck disable=SC2086
      L ${spec#*:} --seconds $((${PROBE_SECONDS:-5} + 2)) --go-file "$B/go" --label "probe-${spec%%:*}" >> "$O/loop.jsonl" &
      probe --label "beside-${spec%%:*}" --wait-for "$B/go"
      wait
    done
  done
  exit 0
fi

B=$(mktemp -d /tmp/host-tier-loop.XXXXXX); trap 'rm -rf "$B"' EXIT
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
{ "$NVCC" --version && g++ -O3 -march=native -Wall -c receipt.cpp -o "$B/receipt.o" \
    && "$NVCC" -O3 -std=c++17 -arch=sm_120 -Xptxas -v -Xcompiler -pthread -o "$B/loop" loop.cu "$B/receipt.o" \
    && "$NVCC" -O3 -std=c++17 -arch=sm_120 -o "$B/pcie" pcie.cu && gcc -O3 -march=native -pthread -Wall -o "$B/dram" dram.c; } \
  > "$O/build.txt" 2>&1 || { cat "$O/build.txt"; exit 4; }
GPU_LEASE_WHO=ra-host-tier gpu-lease "${NGPU:-1}" --wait --max-min 20 -- bash "$PWD/run_loop.sh" leased "$B"
