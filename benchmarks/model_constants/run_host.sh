#!/usr/bin/env bash
# Part B of the model constants (cwd = the source tree): host DRAM on vy-nebius-2 by STREAM (stream.c: copy, triad, a
# read, cached and non-temporal stores), each NUMA node's threads on its own memory, then under one short
# gpu-lease 8 --wait --max-min MAXMIN (default 10) every GPU reading pinned host memory (benchmarks/host_tier/pcie.cu,
# each GPU's buffer on its own node) alone and beside the CPUs' STREAM, timed together (--start-at, both for the same
# SECONDS_EACH, so a pair's PCIe row has no time alone in it). Results go to
# $RESEARCH_RUN_DIR/out: env.txt, stream.jsonl, pcie.jsonl, gpu.txt.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 --campaign cmp-12 \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=W12 -- bash benchmarks/model_constants/run_host.sh
#
# A run in the user slice gets CPUs 0-119 (NUMA 0 = 0-95, NUMA 1 = 96-191; SMT siblings 2c, 2c + 1), so socket 1 has 12
# of its 48 cores here; the per-socket rows say what each socket's cores read. Env: SECONDS_EACH (2), REPS (2), MIB (per
# array per thread, 64; the read's one array a thread then totals 2.3x a socket's 336 MiB of L3 at 12 threads, so the
# read's rows want 512),
# KERNELS (read copy triad copy_nt triad_nt), GPU=0 skips the lease.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
MC=$PWD/benchmarks/model_constants HT=$PWD/benchmarks/host_tier
B=$(mktemp -d /tmp/model-constants-host.XXXXXX); trap 'rm -rf "$B"' EXIT
S=${SECONDS_EACH:-2} R=${REPS:-2} M=${MIB:-64}
export B O S M HT
{ date -u +%FT%TZ; lscpu; numactl -H; taskset -pc $$; grep -H . /proc/self/cgroup; head -40 /proc/meminfo
  cat /sys/kernel/mm/transparent_hugepage/enabled; tail -5 /workspace/pouw/fill/windows; uptime; } > "$O/env.txt" 2>&1
gcc -O3 -march=native -pthread -Wall -o "$B/stream" "$MC/stream.c" > "$O/build.txt" 2>&1 || { cat "$O/build.txt"; exit 4; }
/usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -arch=sm_120 -o "$B/pcie" "$HT/pcie.cu" >> "$O/build.txt" 2>&1 || { cat "$O/build.txt"; exit 4; }
sha256sum "$B/stream" "$B/pcie" "$MC/stream.c" "$HT/pcie.cu" >> "$O/build.txt"

evens() { seq "$1" 2 "$2" | paste -sd,; }       # one thread a core: the even sibling of each
S0C=$(evens 0 94) S0A=0-95 S1C=$(evens 96 118) S1A=96-119 BOTHC=$(evens 0 94),$(evens 96 118)
st() {  # label cpus kernel [extra flags]: one JSON line, with how busy every other CPU was meanwhile
  local label=$1 cpus=$2 k=$3; shift 3
  python3 "$HT/cpuload.py" snap > "$B/a.json"
  line=$("$B/stream" --label "$label" --cpus "$cpus" --kernel "$k" --node local --mib "$M" --seconds "$S" "$@")
  load=$(python3 "$HT/cpuload.py" diff "$B/a.json" --exclude "$cpus")
  echo "${line%\}}, \"load\": $load}" | tee -a "$O/stream.jsonl"
}
export -f st
KERNELS=${KERNELS:-read copy triad copy_nt triad_nt}
for r in $(seq "$R"); do
  for k in $KERNELS; do
    st socket0-core "$S0C" $k; st socket0-all "$S0A" $k; st socket1-core "$S1C" $k; st socket1-all "$S1A" $k
    st both-core "$BOTHC" $k
  done
  for n in 12 24; do  # socket 0 at socket 1's core count and twice it
    for k in triad read; do
      [[ " $KERNELS " == *" $k "* ]] && st socket0-cores$n "$(evens 0 $((2 * n - 2)))" $k
    done
  done
done
[ "${GPU:-1}" = 0 ] && exit 0

# Under the lease: pcie's driver threads on CPUs 0-3 and 96-99 (as ra-host-tier's whole-node window), the CPUs' STREAM on
# one thread a core of the rest.
export PC=0-3,96-99 RC="$(evens 4 94),$(evens 100 118)"
# shellcheck disable=SC2016
gpu-lease 8 --wait --max-min "${MAXMIN:-10}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem \
    --format=csv > "$O/gpu.txt" 2>&1
  date -u +%FT%TZ >> "$O/gpu.txt"
  P() { taskset -c "$PC" "$B/pcie" --seconds 3 --mib 1024 --host-node local "$@"; }
  tag() { sed "s/^{/{\"label\": \"$1\", /" | tee -a "$O/pcie.jsonl"; }
  P --modes h2d --seconds 0.5 | tag gates; [ "${PIPESTATUS[0]}" = 0 ] || { echo "pcie gates failed"; exit 3; }
  P --modes h2d,zcr,d2h | tag alone
  for k in read triad copy_nt; do st readers-alone "$RC" $k; done
  for pair in read:h2d triad:h2d copy_nt:h2d read:zcr; do
    k=${pair%:*} m=${pair#*:}
    t=$(python3 -c "import time; print(time.time() + 10)")
    st "readers-with-$m" "$RC" "$k" --start-at "$t" > "$B/s.json" &
    P --modes "$m" --seconds "$S" --start-at "$t" | tag "with-$k"
    wait; cat "$B/s.json"
  done
  date -u +%FT%TZ >> "$O/gpu.txt"'
