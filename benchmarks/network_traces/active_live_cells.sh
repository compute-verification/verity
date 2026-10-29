#!/usr/bin/env bash
# Several active_live.sh cells back to back in one GPU lease, so their acceptance is read under the same node load, from
# the shipped tree's root, as one preemptible job with 48 GB whose inputs are calibration.json and NAME.trace.json.gz
# and whose outputs are out/*:
#   bash benchmarks/network_traces/active_live_cells.sh NAME="ARGS" ... -- COMMON ARGS
# Each cell writes out/NAME/ with its own vLLM; out/cells.jsonl has each cell's start, end, rc and load averages, and
# out/loadavg.log samples /proc/loadavg every 30 s. A cell gets at most CELL_MINUTES (default 42) and never more than
# what is left of JOB_MINUTES less 4 minutes; one that can't get 10 is skipped. CPUs: a 16-core
# half of the allocation (flock /tmp/nw-sweep-cpus.<h>), 8 of them for vLLM.
set -uo pipefail
OUT=${RESEARCH_RUN_DIR:?run this under research run}/out
mkdir -p "$OUT"
cells=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do cells+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
common=("$@")
c=($(python3 -c "import os;print(*sorted(os.sched_getaffinity(0)))")); mine=("${c[@]}")
if [ ${#c[@]} -ge 32 ]; then
  for h in 0 1; do exec 9>/tmp/nw-sweep-cpus.$h; if flock -n 9; then mine=("${c[@]:$((h*16)):16}"); break; fi; exec 9>&-; done
fi
v=("${mine[@]:0:8}"); cpus=$(IFS=,; echo "${mine[*]}"); export VLLM_CPUS=$(IFS=,; echo "${v[*]}")
echo "cells cpus $cpus vllm $VLLM_CPUS"
(while :; do echo "$(date -u +%FT%TZ) $(cat /proc/loadavg)"; sleep 30; done) > "$OUT/loadavg.log" &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
start=$(date +%s)
for spec in "${cells[@]}"; do
  name=${spec%%=*}; read -r -a args <<< "${spec#*=}"
  left=$(( ${JOB_MINUTES:-140} - ($(date +%s) - start) / 60 - 4 ))
  minutes=$(( left < ${CELL_MINUTES:-42} ? left : ${CELL_MINUTES:-42} ))
  if [ "$minutes" -lt 10 ]; then echo "$name: skipped, $left min left"; continue; fi
  t0=$(date -u +%FT%TZ); l0=$(cut -d' ' -f1-3 /proc/loadavg)
  echo "$name: ${args[*]} (${minutes} min)"
  OUT="$OUT/$name" MINUTES=$minutes taskset -c "$cpus" bash benchmarks/network_traces/active_live.sh "${common[@]}" "${args[@]}"
  rc=$?
  printf '{"cell": "%s", "args": "%s", "minutes": %d, "start": "%s", "end": "%s", "rc": %d, "loadavg_start": "%s", "loadavg_end": "%s", "cpus": "%s"}\n' \
    "$name" "${args[*]}" "$minutes" "$t0" "$(date -u +%FT%TZ)" "$rc" "$l0" "$(cut -d' ' -f1-3 /proc/loadavg)" "$cpus" >> "$OUT/cells.jsonl"
done
