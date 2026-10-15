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
# KERNELS (read copy triad copy_nt triad_nt), GPU=0 skips the lease; REPS=0 skips the CPU phase; READERS, PAIRS and
# QUIET_MAX shape the lease's rows (below).
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

# QUIET_MAX: before the lease, wait until CPUs 0-119 are at most that busy for QUIET_SUSTAIN (3) samples of 3 s, with
# no GPU held and check slots a and b free; inside it, the same with 8 held, else it releases the lease and waits again,
# for up to QUIET_WAIT_MIN (30, whole minutes) in all, then exits 5 without measuring. quiet.jsonl records every gate.
export PC=0-3,96-99 RC="$(evens 4 94),$(evens 100 118)" MC
export READERS=${READERS:-read triad copy_nt} PAIRS=${PAIRS:-read:h2d triad:h2d copy_nt:h2d read:zcr}
export GATE_END=$(( $(date +%s) + 60 * ${QUIET_WAIT_MIN:-30} ))
gate() {  # label held [once]: 0 when quiet (or no QUIET_MAX), 3 when not and time is left, else exit 5; once: 1 sample
  [ -n "${QUIET_MAX:-}" ] || return 0
  local until=$GATE_END sustain=${QUIET_SUSTAIN:-3}
  [ -n "${3:-}" ] && until=0 sustain=1
  python3 "$MC/quiet.py" --max "$QUIET_MAX" --sustain "$sustain" --gpus-held "$2" --until "$until" --exit \
    | sed "s/^{/{\"label\": \"$1\", /" >> "$O/quiet.jsonl"
  local rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] && return 0
  [ "$(date +%s)" -lt "$GATE_END" ] && return 3
  echo "no quiet window within QUIET_WAIT_MIN: not measured"; exit 5
}
export -f gate

# Under the lease: pcie's driver threads on CPUs 0-3 and 96-99 (as ra-host-tier's whole-node window), the CPUs'
# STREAM on one thread a core of the rest. READERS and PAIRS (KERNEL:MODE) name the rows; KERNEL@MIB sets that row's
# MIB.
for try in $(seq 20); do
until gate before 0; do :; done
# shellcheck disable=SC2016
gpu-lease 8 --wait --max-min "${MAXMIN:-10}" -- bash -c '
  gate in-lease 8 once || exit 75
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem \
    --format=csv > "$O/gpu.txt" 2>&1
  date -u +%FT%TZ >> "$O/gpu.txt"
  P() { taskset -c "$PC" "$B/pcie" --seconds 3 --mib 1024 --host-node local "$@"; }
  tag() { sed "s/^{/{\"label\": \"$1\", /" | tee -a "$O/pcie.jsonl"; }
  mib() { [[ $1 == *@* ]] && echo "--mib ${1#*@}"; }
  P --modes h2d --seconds 0.5 | tag gates; [ "${PIPESTATUS[0]}" = 0 ] || { echo "pcie gates failed"; exit 3; }
  P --modes h2d,zcr,d2h | tag alone
  for k in $READERS; do st readers-alone "$RC" ${k%@*} $(mib $k); done
  for pair in $PAIRS; do
    k=${pair%:*} m=${pair#*:}
    t=$(python3 -c "import time; print(time.time() + 10)")
    st "readers-with-$m" "$RC" ${k%@*} $(mib $k) --start-at "$t" > "$B/s.json" &
    P --modes "$m" --seconds "$S" --start-at "$t" | tag "with-$k"
    wait; cat "$B/s.json"
  done
  date -u +%FT%TZ >> "$O/gpu.txt"
  python3 "$MC/quiet.py" | sed "s/^{/{\"label\": \"after\", /" >> "$O/quiet.jsonl"'
rc=$?; [ "$rc" = 75 ] || exit "$rc"
done
