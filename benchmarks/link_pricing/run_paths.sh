#!/usr/bin/env bash
# Node 2's GPU I/O paths (`paths.cu`): throughput and where the bytes land, as one Attempt on verity's CPUs:
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- env PHASE=one taskset -c 48-95 bash benchmarks/link_pricing/run_paths.sh
#
# PHASE=one (1 GPU): copy engines, zero copy, landing in the GPU and in the host, and nsys's DRAM / PCIe sampling of
#   DMA loops into small and large buffers.
# PHASE=two (2 GPUs): peer-to-peer, landing in a GPU written by its peer, nsys sampling of peer writes, and small
#   inbound rings (H2D, peer, SM stores) beside a reader standing in for the receiving GPU's own HBM traffic.
# PHASE=three (2 GPUs): the small inbound rings again, beside a reader of plain loads and one of evict-first loads.
# PHASE=four (2 GPUs): the small inbound rings beside a reader that also drops the ring's lines from L2 unwritten.
# PHASE=five (1 GPU): H2D rings of 1, 2 and 4 MiB beside the plain reader: a ring rewritten faster than L2 turns over.
# GPUS: the indices gpu-lease may take (default 2,3: NUMA node 0's, beside verity's CPUs).
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, gpu.txt, smi.csv, <mode>.jsonl, nsys.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
PHASE=${PHASE:-one}

if [ "${1:-}" = leased ]; then
  S=$2; P=$S/paths
  : "${GPU_LEASE_UUID:?run under gpu-lease}"
  for u in ${GPU_LEASE_UUID//,/ }; do
    nvidia-smi -i "$u" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
  done > "$O/gpu.txt"
  cat "$O/gpu.txt"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory \
    --format=csv,nounits -lms 200 > "$O/smi.csv" 2>&1 &
  SMI=$!
  trap 'kill $SMI 2> /dev/null || true' EXIT
  ns() {  # ns LABEL ARGS...: one nsys-sampled DMA loop, its metrics as a JSON line
    local l=$1; shift
    nsys profile --trace=cuda --gpu-metrics-devices=cuda-visible --gpu-metrics-set=gb20x --gpu-metrics-frequency=10000 \
      --export=sqlite --force-overwrite=true -o "$S/ns_$l" "$P" loop "$@" > "$S/ns_$l.log" 2>&1 ||
      { echo "nsys $l failed"; tail -5 "$S/ns_$l.log"; return 0; }
    grep '"loop"' "$S/ns_$l.log" >> "$O/loops.jsonl" || true
    python3 nsys_metrics.py "$S/ns_$l.sqlite" "$l" >> "$O/nsys.jsonl"
  }
  case $PHASE in
    one)
      "$P" copy --gpu 0 | tee "$O/copy.jsonl"
      "$P" zc --gpu 0 | tee "$O/zc.jsonl"
      "$P" landg --gpu 0 | tee "$O/landg.jsonl"
      "$P" landh --gpu 0 | tee "$O/landh.jsonl"
      for w in smw h2d d2h zcw zcr; do for m in 8 4096; do ns "${w}_${m}" --gpu 0 --what $w --mib $m --seconds 1.5; done; done
      ;;
    two)
      "$P" peer --gpu 0 --peer 1 | tee "$O/peer.jsonl"
      "$P" landg --gpu 1 --peer 0 | tee "$O/landg_peer.jsonl"
      for w in p2pw p2pce; do for m in 8 4096; do ns "${w}_${m}" --gpu 0 --peer 1 --what $w --mib $m --seconds 1.5; done; done
      # small inbound rings beside a 16 GiB reader on the receiving GPU (its own HBM traffic): do they stay in L2?
      ns read_alone --gpu 0 --what none --mib 8 --seconds 1.5 --beside-read 16
      for m in 2 8 32; do ns "h2d_${m}_read" --gpu 0 --what h2d --mib $m --seconds 1.5 --beside-read 16; done
      ns smw_8_read --gpu 0 --what smw --mib 8 --seconds 1.5 --beside-read 16
      for w in p2pw p2pce; do ns "${w}_8_read" --gpu 0 --peer 1 --what $w --mib 8 --seconds 1.5 --beside-read 16; done
      ;;
    three)
      for h in normal stream; do
        r="--seconds 1.5 --beside-read 16 --read-hint $h"
        ns "read_alone_$h" --gpu 0 --what none --mib 8 $r
        for m in 8 32; do ns "h2d_${m}_read_$h" --gpu 0 --what h2d --mib $m $r; done
        ns "smw_8_read_$h" --gpu 0 --what smw --mib 8 $r
        for w in p2pw p2pce; do ns "${w}_8_read_$h" --gpu 0 --peer 1 --what $w --mib 8 $r; done
      done
      ;;
    four)
      r="--seconds 1.5 --beside-read 16 --discard 1"
      ns read_alone_discard --gpu 0 --what none --mib 8 $r
      for m in 8 32; do ns "h2d_${m}_read_discard" --gpu 0 --what h2d --mib $m $r; done
      for w in p2pw p2pce; do ns "${w}_8_read_discard" --gpu 0 --peer 1 --what $w --mib 8 $r; done
      ;;
    five)
      ns read_alone --gpu 0 --what none --mib 8 --seconds 1.5 --beside-read 16
      for m in 1 2 4; do ns "h2d_${m}_read" --gpu 0 --what h2d --mib $m --seconds 1.5 --beside-read 16; done
      ;;
    *) echo "unknown PHASE $PHASE"; exit 2 ;;
  esac
  cat "$O/nsys.jsonl" 2> /dev/null | cut -c1-2000
  exit 0
fi

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
{ "$NVCC" --version | tail -2; } > "$O/build.txt"
"$NVCC" -O3 -std=c++17 -arch=sm_120 -Xptxas -v paths.cu -o "$S/paths" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
sha256sum paths.cu nsys_metrics.py run_paths.sh "$S/paths" >> "$O/build.txt"
echo "SASS lines naming CCTL or DISCARD (the reader's discard.global.L2): $("$(dirname "$NVCC")/cuobjdump" -sass "$S/paths" 2>&1 | grep -c -i -E 'CCTL|DISCARD')" >> "$O/build.txt"
N=2; case $PHASE in one | five) N=1 ;; esac
GPU_LEASE_WHO=ra-link-pricing gpu-lease $N --wait --on "${GPUS:-2,3}" --max-min 25 -- bash "$PWD/run_paths.sh" leased "$S"
