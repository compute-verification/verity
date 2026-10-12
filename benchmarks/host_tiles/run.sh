#!/usr/bin/env bash
# The host CPUs' capacity for registered PoUW tile work: gate bit for bit against the Python reference, then time.
#   taskset -c 48-95 run.sh PHASE...   PHASE = ncp | pearl | latency (cwd = the source tree: research run --cwd source)
# Threads are placed one per physical core (OMP_PLACES=cores) for the scaling sweep, then on both hyperthreads of every core.
# Each phase first records how busy its CPUs were over 2 s (idle_*.txt): a busy CPU is someone else's work, not ours.
set -uxo pipefail
O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
H=benchmarks/host_tiles
B=$(mktemp -d /tmp/host-tiles.XXXXXX); trap 'rm -rf "$B"' EXIT
lscpu > "$O/lscpu.txt"; numactl -H > "$O/numa.txt" 2>&1; taskset -p $$ > "$O/affinity.txt"; nproc > "$O/nproc.txt"
lscpu -p=CPU,CORE,SOCKET,NODE > "$O/cpu_topology.txt"; gcc --version | head -1 > "$O/gcc.txt"
sha256sum $H/*.c $H/*.py $H/run.sh > "$O/sources.sha256"
P=$(nproc); C=$((P / 2))
export OMP_PROC_BIND=close
idle() {  # per-CPU busy fraction over 2 s, for the CPUs this shell may use
  python3 - "$1" <<'PY'
import os, sys, time
cpus = sorted(os.sched_getaffinity(0))
def snap():
    out = {}
    for line in open("/proc/stat"):
        f = line.split()
        if f[0].startswith("cpu") and f[0] != "cpu":
            v = list(map(int, f[1:]))
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out
a = snap(); time.sleep(2); b = snap()
busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in cpus}
with open(sys.argv[1], "w") as f:
    f.write(" ".join(f"{c}:{busy[c]:.2f}" for c in cpus) + "\n")
    f.write(f"max {max(busy.values()):.2f} mean {sum(busy.values()) / len(busy):.3f}\n")
PY
}
gcc -O3 -march=native -fopenmp -o "$B/ncp_amx" $H/ncp_amx.c || exit 4
gcc -O3 -march=native -fopenmp -o "$B/pearl_c_cpu" $H/pearl_c_cpu.c -lm || exit 4
gcc -O3 -march=native -fopenmp -o "$B/stream_read" $H/stream_read.c || exit 4
for phase in "$@"; do
idle "$O/idle_$phase.txt"

if [ "$phase" = ncp ]; then
  ok=1; for i in 1 2 3; do PYTHONPATH=. python3 $H/gate_ncp.py "$B/ncp_amx" >> "$O/gate_ncp.jsonl" || ok=0; done
  [ $ok = 1 ] || { echo "ncp gate failed: no ncp timing" | tee "$O/ncp_gate_failed.txt"; continue; }
  for t in 1 4 12 $C; do OMP_PLACES=cores OMP_NUM_THREADS=$t "$B/stream_read" 4 5 | tee -a "$O/stream.jsonl"; done
  OMP_PLACES=threads OMP_NUM_THREADS=$P "$B/stream_read" 4 5 | tee -a "$O/stream.jsonl"
  # compute-bound: each thread its own 768 KiB Y (k = 1024, n = 256), L2-resident across repeats (two per core with SMT)
  for t in 1 2 4 8 12 16 $C; do
    for m in 16 32 64 128; do
      for ks in 0 128 64 16; do
        OMP_PLACES=cores OMP_NUM_THREADS=$t NCP_REPEAT=64 "$B/ncp_amx" bench $m 3072 256 $((64 * t)) $ks 20 | tee -a "$O/ncp_resident.jsonl"
      done
    done
  done
  for m in 16 32 64 128; do
    for ks in 0 128 64 16; do
      OMP_PLACES=threads OMP_NUM_THREADS=$P NCP_REPEAT=64 "$B/ncp_amx" bench $m 3072 256 $((64 * P)) $ks 20 | tee -a "$O/ncp_resident.jsonl"
    done
  done
  # streaming: 40 weights of k = n = 4096 (Y = 2 GB, six times the LLC), Y read from DRAM every pass
  for t in 1 4 8 12 16 $C; do
    for m in 16 32 64 128; do
      for ks in 0 128 16; do
        OMP_PLACES=cores OMP_NUM_THREADS=$t "$B/ncp_amx" bench $m 12288 4096 40 $ks 5 | tee -a "$O/ncp_stream.jsonl"
      done
    done
  done
  for m in 16 32 64 128; do
    for ks in 0 128 16; do
      OMP_PLACES=threads OMP_NUM_THREADS=$P "$B/ncp_amx" bench $m 12288 4096 40 $ks 5 | tee -a "$O/ncp_stream.jsonl"
    done
  done
fi

if [ "$phase" = pearl ]; then
  ok=1; for i in 1 2 3; do PYTHONPATH=. python3 $H/gate_pearl.py "$B/pearl_c_cpu" "$B/pool.bin" >> "$O/gate_pearl.jsonl" || ok=0; done
  [ $ok = 1 ] || { echo "pearl gate failed: no pearl timing" | tee "$O/pearl_gate_failed.txt"; continue; }
  # 40 weights of k = 4096, n = 6144 (1 GB of codes, streamed): 96 column blocks split evenly over 1 ... 48 threads
  for t in 1 2 4 8 12 16 $C; do
    for m in 1 8 16 32 64 128; do
      for ks in 128 32; do
        w=40; [ $t -lt 4 ] && [ $m -ge 32 ] && w=8
        OMP_PLACES=cores OMP_NUM_THREADS=$t "$B/pearl_c_cpu" bench $m 4096 6144 $w $ks 3 gauss | tee -a "$O/pearl.jsonl"
      done
    done
  done
  for m in 1 8 16 32 64 128; do
    OMP_PLACES=threads OMP_NUM_THREADS=$P "$B/pearl_c_cpu" bench $m 4096 6144 40 128 3 gauss | tee -a "$O/pearl.jsonl"
  done
  for m in 16 128; do
    OMP_PLACES=cores OMP_NUM_THREADS=$C "$B/pearl_c_cpu" bench $m 4096 6144 40 128 3 "file:$B/pool.bin" | tee -a "$O/pearl_data.jsonl"
    OMP_PLACES=cores OMP_NUM_THREADS=$C "$B/pearl_c_cpu" bench $m 4096 6144 4 128 1 uniform | tee -a "$O/pearl_data.jsonl"
  done
fi

if [ "$phase" = latency ]; then
  # The probe on the first core (both its hyperthreads in reach, pinned to the first), tile work on the other cores, then on
  # every CPU including the probe's.  Each kernel runs until the probe finishes, then is stopped by its PID.
  cpus=$(python3 -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0)))))")
  first=${cpus%%,*}; rest=$(echo "$cpus" | cut -d, -f3-); nrest=$(( $(echo "$rest" | tr ',' '\n' | wc -l) / 2 ))
  probe() { taskset -c $first python3 $H/latency_probe.py 20000 "$1" | tee -a "$O/latency.jsonl"; }
  load() {  # load LABEL CPUS PLACES THREADS CMD...: run CMD pinned to CPUS (past its setup) while the probe runs
    local label=$1 set=$2 places=$3 n=$4; shift 4
    OMP_PLACES=$places OMP_NUM_THREADS=$n taskset -c $set "$@" > /dev/null 2>&1 & local pid=$!
    sleep 10; idle "$O/idle_latency_$label.txt"; probe "$label"; kill $pid; wait $pid 2>/dev/null || true
  }
  probe alone
  load ncp_stream_m16_other_cores "$rest" cores $nrest "$B/ncp_amx" bench 16 12288 4096 40 128 100000
  load ncp_resident_m128_other_cores "$rest" cores $nrest env NCP_REPEAT=64 "$B/ncp_amx" bench 128 3072 256 $((64 * nrest)) 128 100000
  load pearl_m16_other_cores "$rest" cores $nrest "$B/pearl_c_cpu" bench 16 4096 6144 40 128 100000 gauss
  load stream_read_other_cores "$rest" cores $nrest "$B/stream_read" 4 100000
  probe alone_again
  load ncp_stream_m16_all_cpus "$cpus" threads $P "$B/ncp_amx" bench 16 12288 4096 40 128 100000
  load pearl_m16_all_cpus "$cpus" threads $P "$B/pearl_c_cpu" bench 16 4096 6144 40 128 100000 gauss
fi
done
