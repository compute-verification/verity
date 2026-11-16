#!/bin/bash
# For each pull request, the ratchet gate's verdict on its own queue merge: ratchet_gate.py MERGE^1 MERGE (what the check's
# ratchet-gate step runs for each `Queue: #N` merge above its base).  usage: gate_runs.sh OUT_DIR '#N' ...
set -u -o pipefail
out=$1; shift; mkdir -p "$out"
for pr in "$@"; do
  m=$(git log --first-parent -1 --format=%H --grep="^Queue: $pr " HEAD)
  [ -z "$m" ] && { echo "$pr no queue merge on HEAD" | tee -a "$out/summary.txt"; continue; }
  timeout 900 python3 adapters/vllm/tools/ratchet_gate.py "$m^1" "$m" --every-rule > "$out/${pr#\#}.txt" 2>&1
  echo "$pr merge ${m:0:12} exit $?" | tee -a "$out/summary.txt"
done
