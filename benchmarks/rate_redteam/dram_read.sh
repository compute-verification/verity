#!/usr/bin/env bash
# dram_read.c on the CPUs the run is given (cwd = the source tree; no GPU): host DRAM read bandwidth at 1 to 48 threads,
# the rate at which a host process could sketch words held in DRAM under a node-level guarantee. Results in
# $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/rate_redteam/dram_read.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
B=$(mktemp -d /tmp/rate-redteam-dram.XXXXXX); trap 'rm -rf "$B"' EXIT
lscpu > "$O/lscpu.txt"; numactl -H > "$O/numa.txt" 2>&1
gcc -O3 -march=native -pthread -o "$B/dram_read" benchmarks/rate_redteam/dram_read.c || exit 4
for cpus in 90 84-95 72-95 48-95; do
  for rep in 1 2 3; do "$B/dram_read" "$cpus" 512 6 | sed "s/^{/{\"cpus\": \"$cpus\", /" | tee -a "$O/dram_read.jsonl"; done
done
