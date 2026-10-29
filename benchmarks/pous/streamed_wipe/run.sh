#!/usr/bin/env bash
# The streamed-label wipe's timing on one RTX PRO 6000 of a GPU node, as one run (cwd = the source tree):
#
#   bash benchmarks/pous/streamed_wipe/run.sh
#
# Outside the lease: build `wipelabels` (refused on a register spill) and the Lean verdict `streamed-wipe-verdict`, which
# judges the session step.  Under `gpu-lease 1 --wait --max-min 20`: `bench.py all` (its docstring lists the steps; a
# failed gate times nothing).  Scratch (the round-trip test's 16 GiB table) is a `mktemp -d` under /dev/shm when it has
# room, else under /tmp, removed on exit.
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, gpu.txt, env.txt, timing.json.
set -euo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
ROOT=$(cd ../../.. && pwd)
export PYTHONPATH=$ROOT:$ROOT/verity/core/difftest/python${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python3}

if [ "${1:-}" = leased ]; then
  S=$2
  UUID=${GPU_LEASE_UUID:?run under gpu-lease}
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$UUID}
  NAME=$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader | head -1)
  nvidia-smi -i "$UUID" --query-gpu=index,uuid,name,pci.bus_id,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv > "$O/gpu.txt"
  case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $UUID is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
  { nvidia-smi -i "$UUID" -q -d CLOCK,PERFORMANCE,POWER; lscpu | grep -E "Model name|^CPU\(s\)|NUMA"; nproc; uptime;
    df -h /dev/shm /tmp; which ib_write_bw iperf3 || true; ethtool eth0 2>/dev/null | grep -E "Speed|Duplex" || true; } \
    > "$O/env.txt" 2>&1
  if [ "$(df --output=avail -B1G /dev/shm | tail -1 | tr -d ' ')" -ge 24 ]; then BASE=/dev/shm; else BASE=/tmp; fi
  D=$(mktemp -d -p "$BASE" streamed-wipe.XXXXXX); trap 'rm -rf "$D"' EXIT
  "$PY" bench.py all --bin "$S/wipelabels" --scratch "$D" --out "$O"
  exit 0
fi

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
FLAGS=(-O3 -std=c++17 -arch=sm_120 -Xptxas -v)
{ "$NVCC" --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
"$NVCC" "${FLAGS[@]}" labels.cu -o "$S/wipelabels" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
if grep -E "[1-9][0-9]* bytes spill" "$O/build.txt"; then echo "REFUSED: a kernel spills"; exit 3; fi
sha256sum labels.cu bench.py "$S/wipelabels" >> "$O/build.txt"
(cd "$ROOT/verity/core/devices/memory_challenger/lean" && lake build streamed-wipe-verdict) >> "$O/build.txt" 2>&1 \
  || { tail -30 "$O/build.txt"; echo "REFUSED: the Lean verdict doesn't build"; exit 3; }

rc=0
GPU_LEASE_WHO=memory-accounting gpu-lease 1 --wait --max-min 20 -- bash "$PWD/run.sh" leased "$S" || rc=$?
[ -f "$O/timing.json" ] && "$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])).get('summary'), indent=1))" "$O/timing.json"
exit $rc
