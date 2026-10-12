#!/usr/bin/env bash
# The host tier's honest loop against the adversary's best read on the same CPUs and GPUs, and what the loop costs a
# vLLM-like host process, as one Attempt (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- env NGPU=1 taskset -c 48-95 bash benchmarks/host_tier/run_loop.sh
#
# Outside the lease: build loop.cu (+ receipt.cpp), pcie.cu and dram.c. Under `gpu-lease $NGPU --wait [--on GPU_ON]
# --max-min 20`: the loop's gates against reference.py for KS 16, 4, 1, the
# copy-engine path and the 1 MiB chunk path (a failed gate times nothing, exit 3); then, REPS times, the CPUs' receipts
# alone on every worker CPU (KS 1 by groups; KS 1 and 4 by chunks; KS 1 by chunks prefetching twice as far), the
# adversary's CPU read alone and beside the GPUs' copy engines (loop.cu), the full loop (copy engine: KS 4 by groups,
# KS 1 and 4 by chunks; zero copy: KS 1 by chunks), and the adversary's CPU read timed with each GPU's copy engines
# (dram.c and pcie.cu, as run_pcie.sh); then hostproc.py alone and beside the loop, the adversary, and the GPUs'
# receipts alone. CPU receipts hash their leaves on the SHA extensions (SHA=ni) unless SHA=c.
# All host-tier memory is on NUMA node GPU_NODE (0), the node of the worker CPUs: one socket's DRAM.
# Env: NGPU (1), REPS (4), SECONDS_EACH (3), LOOP_GRID (32), WORKERS (52-95), HOST_CPUS (GPU driver threads, 48,49),
#      PROBE_CPU (50), PROBE_SECONDS (5), PROBE_REPS (2), GPU_NODE (0), GPU_ON, SHA (ni).
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, gpu.txt, gate*/, gates.jsonl, loop.jsonl, pcie.jsonl, dram.jsonl,
# hostproc.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"

if [ "${1:-}" = leased ]; then
  B=$2
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem \
    --format=csv > "$O/gpu.txt" 2>&1
  S=${SECONDS_EACH:-3} G=${LOOP_GRID:-32} W=${WORKERS:-52-95} H=${HOST_CPUS:-48,49} PR=${PROBE_CPU:-50} N=${NGPU:-1}
  SHA=${SHA:-ni}
  L() { "$B/loop" --seconds "$S" --gpu-mib 2048 --cpu-mib 256 --adv-mib 256 --gpu-node "${GPU_NODE:-0}" --host-cpus "$H" "$@"; }
  one() {
    python3 cpuload.py snap > "$B/a.json"
    line=$(L "$@")
    load=$(python3 cpuload.py diff "$B/a.json" --exclude 48-95)
    echo "${line%\}}, \"load\": $load, \"cuda_visible\": \"$CUDA_VISIBLE_DEVICES\"}" | tee -a "$O/loop.jsonl"
  }
  gate() {  # gate NAME FLAGS...: the loop's first receipts against reference.py; a failed gate times nothing
    mkdir -p "$O/$1"
    L --gate "$O/$1" --cpus "${W%%[-,]*}" "${@:2}" || { echo "gate run $1 failed"; exit 3; }
    python3 reference.py "$O/$1/gate.json" | sed "s/^{/{\"name\": \"$1\", /" | tee -a "$O/gates.jsonl"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "gate $1 failed"; exit 3; }
  }
  for k in 16 4 1; do gate "gate$k" --ks "$k" --sha "$SHA"; done
  gate gate4-dma --ks 4 --sha "$SHA" --gpu-mode dma
  for k in 4 1; do gate "gate$k-chunk" --ks "$k" --sha "$SHA" --chunk 1; done
  for r in $(seq "${REPS:-4}"); do
    one --label cpu --gpus 0 --cpus "$W" --ks 1 --sha "$SHA"
    for k in 1 4; do one --label cpu --gpus 0 --cpus "$W" --ks "$k" --sha "$SHA" --chunk 1; done
    one --label cpu --gpus 0 --cpus "$W" --ks 1 --sha "$SHA" --chunk 1 --pf 2048
    one --label adv --gpus 0 --adv-cpus "$W"
    one --label gpu+adv --gpus "$N" --grid 4 --gpu-mode dma --adv-cpus "$W"
    one --label honest --gpus "$N" --grid "$G" --gpu-mode dma --cpus "$W" --ks 4 --sha "$SHA"
    for k in 1 4; do one --label honest --gpus "$N" --grid "$G" --gpu-mode dma --cpus "$W" --ks "$k" --sha "$SHA" --chunk 1; done
    one --label honest --gpus "$N" --grid "$G" --cpus "$W" --ks 1 --sha "$SHA" --chunk 1
    # the adversary's CPU read beside the GPUs' copy engines, timed together
    D() { "$B/dram" --cpus "$W" --node "${GPU_NODE:-0}" --kernel multi --thp 1 --mib 256 --seconds "$S" "$@"; }
    D --label cpu-alone | tee -a "$O/dram.jsonl"
    for m in h2d; do
      t=$(python3 -c 'import time; print(time.time() + 8)')
      D --label "cpu-with-$m" --start-at "$t" > "$B/d.json" &
      taskset -c "$H" "$B/pcie" --seconds "$S" --mib 1024 --host-node "${GPU_NODE:-0}" --modes "$m" --start-at "$t" \
        | sed "s/^{/{\"label\": \"with-cpu-$m\", \"cuda_visible\": \"$CUDA_VISIBLE_DEVICES\", /" | tee -a "$O/pcie.jsonl"
      wait; tee -a "$O/dram.jsonl" < "$B/d.json"
    done
  done
  probe() { taskset -c "$PR" python3 hostproc.py --seconds "${PROBE_SECONDS:-5}" "$@" | tee -a "$O/hostproc.jsonl"; }
  for r in $(seq "${PROBE_REPS:-2}"); do
    probe --label alone
    for spec in "honest:--gpus $N --grid $G --gpu-mode dma --cpus $W --ks 1 --sha $SHA --chunk 1" "adv:--gpus 0 --adv-cpus $W" \
      "gpu:--gpus $N --grid $G --gpu-mode dma"; do
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
# GPU_ON (e.g. 0,1,2,3, GPU_NODE 0's GPUs) only when those are free: a waiting --on request holds every later waiter
# back, even from GPUs outside its list
GPU_LEASE_WHO=ra-host-tier gpu-lease "${NGPU:-1}" --wait ${GPU_ON:+--on "$GPU_ON"} --max-min 20 -- bash "$PWD/run_loop.sh" leased "$B"
