#!/usr/bin/env bash
# Round 2 of the census: host AMX (int8, bf16) on every physical core this run's cgroup allows, per core against 1 and
# 16 cores, alone and beside dense FP8 GEMMs (gemm_fp8.cu) on the GPUs of one `gpu-lease 2`, and as a share of the
# node's FP8 tensor capacity (census.py --amx). Every host process runs under nice 19, one AMX thread a physical core
# (the lowest CPU of each), each run a few seconds; before each run outside the lease the script waits (from a budget of
# GATE_S seconds) while another GPU lease is held or more than BUSY_MAX CPUs are busy, and records what it could not
# wait out. PHASES (default "A B C") picks alone, beside the GEMMs, and alone again. Inside a research run:
#   bash benchmarks/pouw/census_sm120/host_amx.sh
# writes $OUT/amx.jsonl (host.c amx), freq.jsonl (host.c freq), gemm-<i>.jsonl, smi.csv, timeline.log, amx-summary.json,
# amx.md and the cgroup, topology and lease records.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?}/out}
mkdir -p "$OUT"
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SEC=${SEC:-4}
GATE_S=${GATE_S:-900}
BUSY_MAX=${BUSY_MAX:-4}
PHASES=${PHASES:-A B C}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
J="$OUT/amx.jsonl"
F="$OUT/freq.jsonl"
log() { echo "$(date -u +%H:%M:%S.%3NZ) $*" | tee -a "$OUT/timeline.log"; }

gcc -O3 -march=native -fopenmp -mamx-tile -mamx-int8 -mamx-bf16 -o "$W/host" "$SRC/host.c" -lpthread >"$OUT/build-host.log" 2>&1 \
  || { echo "host build failed" >&2; exit 1; }
H="$W/host"
GEMM="$W/gemm_fp8"
/usr/local/cuda-13.0/bin/nvcc -O3 -arch=sm_120a -o "$GEMM" "$SRC/gemm_fp8.cu" -lcublasLt >"$OUT/build-gemm.log" 2>&1 \
  || { echo "gemm build failed" >&2; GEMM=; }

{
  grep -E '^(Cpus|Mems)_allowed_list' /proc/self/status
  echo "cgroup: $(cat /proc/self/cgroup)"
  cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
  while [ "$cg" != /sys/fs ] && [ -n "$cg" ]; do
    echo "$cg cpuset.cpus=$(cat "$cg/cpuset.cpus" 2>/dev/null) effective=$(cat "$cg/cpuset.cpus.effective" 2>/dev/null)"
    cg=${cg%/*}
  done
  for s in user.slice system.slice provers.slice vy.slice _devices.slice pous.slice; do
    echo "$s: $(systemctl show -p AllowedCPUs -p EffectiveCPUs "$s" 2>&1 | tr '\n' ' ')"
  done
  nproc
  grep -m1 'model name' /proc/cpuinfo
  ls /sys/devices/system/cpu/cpu0/cpufreq/ 2>&1 | tr '\n' ' '; echo
  echo "powercap: $(ls /sys/class/powercap 2>&1 | tr '\n' ' ')"
  date; date -u
} >"$OUT/cgroup.txt" 2>&1
lscpu >"$OUT/host-lscpu.txt" 2>&1
lscpu -p=CPU,CORE,SOCKET,NODE >"$OUT/host-lscpu-p.txt" 2>&1

# One CPU (the lowest) a physical core of the allowed set, by socket; the 1- and 16-core sets as round 1 (from CPU 16 on
# socket 0); SMT = every allowed CPU; SIB0/SIB1 = the SMT siblings of the first core used on each socket.
allowed=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')
eval "$(python3 - "$allowed" <<'PY'
import subprocess, sys
ok = set()
for part in sys.argv[1].split(","):
    a, _, b = part.partition("-")
    ok.update(range(int(a), int(b or a) + 1))
cores = {}
for line in subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET"], capture_output=True, text=True).stdout.splitlines():
    if line.startswith("#"):
        continue
    cpu, core, sock = (int(x) for x in line.split(","))
    if cpu in ok:
        cores.setdefault((sock, core), []).append(cpu)
first = sorted(min(v) for v in cores.values())
sock = {min(v): s for (s, _), v in cores.items()}
sib = {min(v): sorted(v)[1] for v in cores.values() if len(v) > 1}
s0 = [c for c in first if sock[c] == 0]
s1 = [c for c in first if sock[c] == 1]
from16 = [c for c in s0 if c >= 16] or s0
j = lambda xs: ",".join(map(str, xs))
print(f"ALL={j(first)} S0={j(s0)} S1={j(s1)} ONE={from16[0]} SIXTEEN={j(from16[:16])} SMT={j(sorted(ok))}")
print(f"SIB0={sib.get(from16[0], '')} SIB1={sib.get(s1[0], '') if s1 else ''}")
PY
)"
echo "{\"allowed\":\"$allowed\",\"all\":\"$ALL\",\"s0\":\"$S0\",\"s1\":\"$S1\",\"one\":\"$ONE\",\"sixteen\":\"$SIXTEEN\",\"sib0\":\"$SIB0\",\"sib1\":\"$SIB1\"}" \
  >"$OUT/amx-cpus.json"
n() { awk -F, '{print NF}' <<<"$1"; }

# CPU-equivalents busy over 1 s, across every CPU of the node (/proc/stat)
busy() {
  python3 - <<'PY'
import time
def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = list(map(int, f[1:9]))
            out[f[0]] = (sum(v), v[3] + v[4])
    return out
a = snap(); time.sleep(1); b = snap()
print(round(sum(1 - (b[k][1] - a[k][1]) / max(1, b[k][0] - a[k][0]) for k in b), 2))
PY
}
others() { gpu-lease status 2>/dev/null | grep -E '^[0-9] held' | grep -v "run=${RESEARCH_RUN_ID:-none} " | cut -c1-110 || true; }
# Before each AMX run: wait, from a budget of GATE_S seconds for the whole run, while another GPU lease is held or more
# than BUSY_MAX CPUs are busy; echoes the busy reading the run starts at.
echo "$GATE_S" >"$W/gate_left"
quiet() {
  local b o t=0 left
  left=$(cat "$W/gate_left")
  [ -n "${NOGATE:-}" ] && left=0
  while b=$(busy) && o=$(others | tr '\n' ';') && { [ -n "$o" ] || awk "BEGIN {exit !($b > $BUSY_MAX)}"; } \
    && [ "$left" -gt 0 ]; do
    [ "$t" -eq 0 ] && echo "$(date -u +%H:%M:%S.%3NZ) gate: waiting (busy=$b, other leases: $o)" >>"$OUT/timeline.log"
    sleep 14
    t=$((t + 15)) left=$((left - 15))
  done
  [ -z "${NOGATE:-}" ] && echo "$left" >"$W/gate_left"
  [ "$t" -gt 0 ] && echo "$(date -u +%H:%M:%S.%3NZ) gate: waited ${t}s, ${left}s left; other leases: $o" >>"$OUT/timeline.log"
  echo "$b"
}
amx() { # CPUS TAG VARIANTS [SECONDS]
  log "amx $2 $3 cpus=$(n "$1") busy=$(quiet)"
  nice -n 19 taskset -c "$1" env OMP_NUM_THREADS="$(n "$1")" OMP_PROC_BIND=close OMP_PLACES=threads \
    "$H" amx "$(n "$1")" "${4:-$SEC}" "$2" "$3" >>"$J" 2>>"$OUT/host.err"
}
freq() { # CPUS SECONDS TAG
  nice -n 19 taskset -c "$1" env OMP_NUM_THREADS="$(n "$1")" OMP_PROC_BIND=close OMP_PLACES=threads \
    "$H" freq "$(n "$1")" "$2" "$3" >>"$F" 2>>"$OUT/host.err"
}
# amx beside freq on the SMT siblings SIBS of cores it uses: the clock of a core running AMX
amx_sib() { # CPUS TAG SIBS
  local secs p
  secs=$(awk "BEGIN {print 2 * $SEC}")
  freq "$3" "$secs" "$2" &
  p=$!
  amx "$1" "$2" resident
  wait "$p"
}

phase_host() { # PREFIX: every core set alone, every variant (the PREFIX'd runs are in phase B, beside the GEMMs)
  local p=$1 v=${2:-resident,l1,l2}
  amx "$ONE" "$p.1core" "$v"
  amx "$SIXTEEN" "$p.16core" "$v"
  amx "$S0" "$p.s0" "$v"
  [ -n "$S1" ] && amx "$S1" "$p.s1" "$v"
  amx "$ALL" "$p.all" "$v"
}

others >"$OUT/leases-start.txt"
if [[ " $PHASES " == *" A "* ]]; then
log "phase A: alone, cpus all=$(n "$ALL") s0=$(n "$S0") s1=$(n "$S1")"
freq "$ONE" 2 ".1core.scalar"
amx "$ONE" ".1core" resident,l1,l2
amx "$SIXTEEN" ".16core" resident,l1,l2
amx "$S0" ".s0" resident,l1,l2
[ -n "$S1" ] && amx "$S1" ".s1" resident,l1,l2
amx "$ALL" ".all" resident,l1,l2
amx "$SMT" ".smt" resident
freq "$ALL" 2 ".all.scalar"
amx_sib "$ONE" ".1core.sibfreq" "$SIB0"
amx_sib "$ALL" ".all.sibfreq" "$SIB0${SIB1:+,$SIB1}"
amx "$ALL" ".all.sustained" resident 20
fi

if [[ " $PHASES " != *" B "* ]]; then
  :
elif [ -n "$GEMM" ]; then
  log "phase B: waiting for gpu-lease 2"
  cat >"$W/leased.sh" <<'EOF'
set -uo pipefail
log() { echo "$(date -u +%H:%M:%S.%3NZ) $*" | tee -a "$OUT/timeline.log"; }
log "leased: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU_LEASE_UUID=$GPU_LEASE_UUID $(grep Cpus_allowed_list /proc/self/status | tr '\t' ' ') cgroup=$(cut -d: -f3 /proc/self/cgroup)"
gpu-lease status 2>/dev/null | grep -E '^[0-9] held' | cut -c1-110 >"$OUT/leases-phaseB.txt"
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.max.sm,power.limit --format=csv \
  >"$OUT/gpus.csv" 2>&1
for b in $(nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=pci.bus_id --format=csv,noheader); do
  b=$(tr 'A-Z' 'a-z' <<<"${b#0000}"); echo "$b numa_node=$(cat /sys/bus/pci/devices/0000${b#0000}/numa_node 2>&1)"
done >>"$OUT/gpus.csv"
nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,index,uuid,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
  --format=csv -lms 250 >"$OUT/smi.csv" 2>&1 &
smi=$!
export GEMM_STOP="$W/gemm.stop"
IFS=, read -ra devs <<<"$CUDA_VISIBLE_DEVICES"
pids=()
for i in "${!devs[@]}"; do
  CUDA_VISIBLE_DEVICES=${devs[$i]} taskset -c "$((2 * i + 1))" "$GEMM" 8192 900 100 "gpu$i" >"$OUT/gemm-$i.jsonl" 2>"$OUT/gemm-$i.err" &
  pids+=($!)
done
log "gemms started on ${#devs[@]} GPUs; alone for 15 s"
sleep 15
. "$W/funcs.sh"
NOGATE=1
phase_host ".gpu2" resident,l2
amx_sib "$ALL" ".gpu2.all.sibfreq" "$SIB0${SIB1:+,$SIB1}"
amx "$ALL" ".gpu2.all.sustained" resident 20
log "amx done; gemms alone for 15 s"
sleep 15
touch "$GEMM_STOP"
wait "${pids[@]}"
kill "$smi"
log "phase B done"
EOF
  declare -f log busy others quiet n amx freq amx_sib phase_host >"$W/funcs.sh"
  echo "H=$H J=$J F=$F SEC=$SEC BUSY_MAX=$BUSY_MAX ONE=$ONE SIXTEEN=$SIXTEEN S0=$S0 S1=$S1 ALL=$ALL SIB0=$SIB0 SIB1=$SIB1" >>"$W/funcs.sh"
  OUT="$OUT" W="$W" GEMM="$GEMM" gpu-lease 2 --wait --max-min 20 -- bash "$W/leased.sh" >"$OUT/lease.log" 2>&1
  log "lease exit $?"
else
  log "phase B skipped: no GEMM binary (build-gemm.log)"
fi

others >"$OUT/leases-end.txt"
if [[ " $PHASES " == *" C "* ]]; then
  log "phase C: alone again"
  amx "$ALL" ".post.all" resident
  amx "$ONE" ".post.1core" resident
fi
python3 "$SRC/census.py" --amx "$OUT"
log "done"
