#!/usr/bin/env bash
# The certified PCIe stream (`stream.py`, `stream/v1`) on one RTX PRO 6000 of vy-nebius-2, as one Attempt (cwd = the source
# tree), on verity's CPUs of NUMA node 0 and a GPU attached to it:
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/bandwidth/run_stream.sh
#
# Outside the lease: build `bwstream` (stream.cu with tagger.cu, and stream_host.cpp by g++; refused on any register spill),
# regenerate the pinned stream vectors and check them, and compute with the core reference the full store's expected roots.
# Under `gpu-lease 1 --wait --on $GPUS --max-min 25`: `bwstream --mode gates` (the host generators against the keystream,
# the device absorb and the host reference against the pinned digests and group nodes, a control without the copies that
# must not match, the tag pass's root); a failed gate times nothing (exit 3).  Then the timings, each its own JSON: link, hbm,
# gen, pipe, absorb, window.
# Env: BLOCKS (default 524,288), SEED (hex, default 00..1f), MODES (default "link hbm gen pipe absorb interfere window";
# also fold),
# REPS (7), WINDOWS (20), SIZES_MIB (16,32,48,56,60,64), PIECE (bytes, 1 MiB), GEN_THREADS (16), WINDOWS_ALSO (more window
# configurations, `suffix:args;...`), GPUS (0,1,2,3: NUMA node 0's), WORKERS (the reference's processes, 32), TIMED,
# MPS (1: an MPS daemon for WINDOWS_ALSO's `--mps-dir @MPS@`), MPS_CPUS (the daemon's and its server's CPUs).
# PHYS (1: one thread of each idle physical core of the affinity, picked once the GPU is leased).
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, host.txt, expect.json, gpu.txt, env.txt, smi.csv, <mode>.json for each mode.
set -euo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
export PYTHONPATH=$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python3}

if [ "${1:-}" = leased ]; then
  S=$2
  UUID=${GPU_LEASE_UUID:?run under gpu-lease}
  NAME=$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader | head -1)
  nvidia-smi -i "$UUID" --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv > "$O/gpu.txt"
  case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $UUID is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
  BUS=$(nvidia-smi -i "$UUID" --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr 'A-F' 'a-f' | sed 's/^0000\(....\)/\1/')
  { nvidia-smi -i "$UUID" -q -d CLOCK,PERFORMANCE,POWER,MEMORY; nvidia-smi topo -m;
    for f in current_link_speed current_link_width max_link_speed max_link_width numa_node local_cpulist; do
      echo "sysfs $f: $(cat /sys/bus/pci/devices/${BUS,,}/$f 2> /dev/null || echo '?')"; done
    # the payload sizes bound the link's payload rate below its signalling rate: the GPU and the port above it
    for d in "${BUS,,}" "$(basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/${BUS,,}")")")"; do
      echo "lspci $d:"; { sudo -n lspci -vvv -s "$d" 2> /dev/null || lspci -vvv -s "$d"; } |
        grep -E 'MaxPayload|MaxReadReq|LnkCap:|LnkSta:|access denied'; done; } > "$O/env.txt" 2>&1
  nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,pcie.link.gen.current \
    --format=csv,nounits -lms 100 > "$O/smi.csv" &
  SMI=$!
  trap 'kill $SMI 2> /dev/null || true' EXIT
  # PHYS=1: run on one thread of each physical core whose threads are all in this affinity and were idle for a second.
  # A generator beside a busy sibling shares its VAES units, and one slow generator stalls the frontier, which advances
  # only over slots written in order.  MPS_CPUS then defaults to the last two picked.
  if [ -n "${PHYS:-}" ]; then
    pick=$("$PY" -c '
import os, time
def stat():
    return {int(l.split()[0][3:]): [int(x) for x in l.split()[1:]] for l in open("/proc/stat") if l[3].isdigit()}
def cpus(s):
    return [c for p in s.strip().split(",") for c in range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1)]
a = stat(); time.sleep(1); b = stat()
busy = {c: 1 - (b[c][3] + b[c][4] - a[c][3] - a[c][4]) / max(1, sum(b[c]) - sum(a[c])) for c in a}
aff = os.sched_getaffinity(0)
sib = {c: cpus(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read()) for c in aff}
print(",".join(str(c) for c in sorted(aff) if c == min(sib[c]) and all(s in aff and busy[s] < 0.1 for s in sib[c])))')
    [ -n "$pick" ] || { echo "REFUSED: PHYS=1 but no idle physical core"; exit 3; }
    taskset -pc "$pick" $$ > /dev/null
    MPS_CPUS=${MPS_CPUS:-$(echo "$pick" | tr ',' '\n' | tail -n 2 | paste -sd,)}
    echo "PHYS: $pick (MPS on ${MPS_CPUS:-})" | tee -a "$O/host.txt"
  fi
  common=(--out "$O" --blocks "${BLOCKS:-524288}" --reps "${REPS:-7}" --windows "${WINDOWS:-20}"
          --sizes-mib "${SIZES_MIB:-16,32,48,56,60,64}" --piece "${PIECE:-1048576}" --gen-threads "${GEN_THREADS:-16}")
  rc=0
  "$S/bwstream" --mode gates "${common[@]}" --vectors tests/stream_vectors.json --expect "$O/expect.json" || rc=$?
  [ $rc = 0 ] || { echo "gates exited $rc: nothing is timed"; exit $rc; }
  for m in ${MODES:-link hbm gen pipe absorb interfere window}; do
    "$S/bwstream" --mode "$m" "${common[@]}" || { rc=$?; echo "mode $m exited $rc"; exit $rc; }
  done
  # MPS=1: an MPS control daemon of this run's own (its pipe and logs under $S, so no other process finds it) for the
  # leased GPU, quit on exit; `@MPS@` in WINDOWS_ALSO's args is its pipe directory (`--mps-dir @MPS@`).
  if [ -n "${MPS:-}" ]; then
    command -v nvidia-cuda-mps-control > /dev/null || { echo "REFUSED: MPS=1 but no nvidia-cuda-mps-control"; exit 3; }
    export MPSDIR=$S/mps-pipe MPSLOG=$S/mps-log; mkdir -p "$MPSDIR" "$MPSLOG"
    CUDA_VISIBLE_DEVICES=$UUID CUDA_MPS_PIPE_DIRECTORY=$MPSDIR CUDA_MPS_LOG_DIRECTORY=$MPSLOG \
      ${MPS_CPUS:+taskset -c "$MPS_CPUS"} nvidia-cuda-mps-control -d
    trap 'echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPSDIR nvidia-cuda-mps-control; sleep 1; cp "$MPSLOG"/*.log "$O"/ 2> /dev/null;
          nvidia-smi -i "$UUID" --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/mps_after.txt" 2>&1;
          kill $SMI 2> /dev/null || true' EXIT
  fi
  # WINDOWS_ALSO: more window configurations, `;`-separated, each `suffix:args` (e.g. `_graph:--path graph`)
  IFS=';' read -ra extra <<< "${WINDOWS_ALSO:-}"
  for e in "${extra[@]}"; do
    args=${e#*:}; args=${args//@MPS@/${MPSDIR:-}}
    # an MPS client names its GPU within the server's visible set, where the leased GPU is device 0
    cvd=$CUDA_VISIBLE_DEVICES; [[ $args == *--mps-dir* ]] && cvd=0
    # each entry is gated on its own checks, so one that fails ends only itself; the run exits with the first failure
    CUDA_VISIBLE_DEVICES=$cvd "$S/bwstream" --mode window "${common[@]}" --suffix "${e%%:*}" $args ||
      { r=$?; echo "window ${e%%:*} exited $r"; [ "$rc" = 0 ] && rc=$r; }
  done
  exit $rc
fi

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
FLAGS=(-O3 -std=c++17 -arch=sm_120 -Xptxas -v)
HOSTFLAGS=(-O3 -mavx512f -mavx512bw -mvaes -maes)
{ "$NVCC" --version; g++ --version | head -1; echo "flags: ${FLAGS[*]}; host: ${HOSTFLAGS[*]}"; } > "$O/build.txt"
g++ "${HOSTFLAGS[@]}" -c stream_host.cpp -o "$S/stream_host.o" >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
"$NVCC" "${FLAGS[@]}" stream.cu "$S/stream_host.o" -o "$S/bwstream" -ldl >> "$O/build.txt" 2>&1 || { tail -30 "$O/build.txt"; exit 3; }
if grep -E "[1-9][0-9]* bytes spill" "$O/build.txt"; then echo "REFUSED: a kernel spills"; exit 3; fi
sha256sum stream.cu tagger.cu ../../verity/ml/kernels/sha512.cuh stream_host.cpp stream.py tests/stream_vectors.json "$S/bwstream" >> "$O/build.txt"
{ lscpu; echo; numactl -H 2> /dev/null || true; echo "affinity: $(taskset -pc $$)";
  grep -o -w -E 'vaes|avx512f|avx512bw|aes|vpclmulqdq' /proc/cpuinfo | sort | uniq -c;
  ls /usr/lib/x86_64-linux-gnu/libcrypto.so.3 2> /dev/null || echo "no libcrypto.so.3"
  echo "hyperthread siblings: $(sort -u -t, -n /sys/devices/system/cpu/cpu*/topology/thread_siblings_list | tr '\n' ' ')"
  # each CPU's busy share over one second, since a sibling's load halves a generator core's VAES rate
  a=$(grep '^cpu[0-9]' /proc/stat); sleep 1; b=$(grep '^cpu[0-9]' /proc/stat)
  paste <(echo "$a") <(echo "$b") | awk '{n = NF / 2; i0 = $5 + $6; i1 = $(n + 5) + $(n + 6); t0 = 0; t1 = 0;
    for (k = 2; k <= n; k++) { t0 += $k; t1 += $(n + k) } printf "%s:%.0f%% ", $1, 100 * (1 - (i1 - i0) / (t1 - t0 + 1e-9)) }
    END { print "" }'; } > "$O/host.txt" 2>&1

"$PY" stream.py "$S/stream_vectors.json"
cmp "$S/stream_vectors.json" tests/stream_vectors.json || { echo "REFUSED: tests/stream_vectors.json is not stream.py's"; exit 3; }
"$PY" reference.py --blocks "${BLOCKS:-524288}" --seed "${SEED:-000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f}" \
  --workers "${WORKERS:-32}" --out "$O/expect.json" > /dev/null

GPU_LEASE_WHO=${GPU_LEASE_WHO:-ra-pcie-exhaust} gpu-lease 1 --wait --on "${GPUS:-0,1,2,3}" ${TIMED:+--timed} --max-min 25 -- \
  bash "$PWD/run_stream.sh" leased "$S"
