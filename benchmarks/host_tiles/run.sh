#!/usr/bin/env bash
# The host CPUs' capacity for registered PoUW tile work: gate bit for bit against the Python reference, then time.
#   run.sh ncp | pearl | latency      (cwd = the source tree: research run --cwd source; results in $OUT)
# Threads are placed one per physical core (OMP_PLACES=cores) for the scaling sweep, then on both hyperthreads of every core.
set -uxo pipefail
O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
H=benchmarks/host_tiles
B=$(mktemp -d /tmp/host-tiles.XXXXXX); trap 'rm -rf "$B"' EXIT
lscpu > "$O/lscpu.txt"; numactl -H > "$O/numa.txt" 2>&1; taskset -p $$ > "$O/affinity.txt"; nproc > "$O/nproc.txt"
lscpu -p=CPU,CORE,SOCKET,NODE > "$O/cpu_topology.txt"; gcc --version | head -1 > "$O/gcc.txt"
sha256sum $H/*.c $H/*.py $H/run.sh > "$O/sources.sha256"
P=$(nproc); C=$((P / 2))
export OMP_PROC_BIND=close
phase=${1:?ncp|pearl|latency}

if [ "$phase" = ncp ]; then
  gcc -O3 -march=native -fopenmp -o "$B/ncp_amx" $H/ncp_amx.c || exit 4
  gcc -O3 -march=native -fopenmp -o "$B/stream_read" $H/stream_read.c || exit 4
  for i in 1 2 3; do PYTHONPATH=. python3 $H/gate_ncp.py "$B/ncp_amx" >> "$O/gate_ncp.jsonl" || { echo "gate failed"; exit 5; }; done
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
