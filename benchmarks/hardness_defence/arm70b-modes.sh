#!/usr/bin/env bash
# Llama-3.1-70B's perplexity under several serving specs at one codes map, on node 2: WikiText-2 and FineWeb-Edu val's
# two halves (as arm70b.sh), the specs in leases of 2 GPUs under arm.sh's 30 minutes, at most $HD_WT specs a WikiText-2
# lease (default 5) and $HD_FW a FineWeb-Edu half's (default 2), each lease taking every matrix's c from CALIB (qat.py
# --calib). Lease NAME-SET-I serves its first spec as --serve and the rest as --eval-modes.
#   bash benchmarks/hardness_defence/arm70b-modes.sh NAME TARGETS CALIB SPEC...
set -euo pipefail
name=$1 targets=$2 calib=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd)
common=(--model "${HD_70B:-/dev/shm/compute-research-hd/Meta-Llama-3.1-70B}" --device-map auto --tokens 0 --tasks ""
  --target 0.02 --rotate --targets "$targets" --calib "$calib")
pids=()
lease() {  # SET PER QAT_ARGS... : one lease per PER specs
  local set=$1 per=$2 i=0
  shift 2
  local specs=("${SPECS[@]}")
  while [ ${#specs[@]} -gt 0 ]; do
    local chunk=("${specs[@]:0:$per}") rest
    specs=("${specs[@]:$per}")
    rest=$(IFS=,; echo "${chunk[*]:1}")
    bash "$here/arm.sh" "$name-$set-$i" 2 "${common[@]}" "$@" --serve "${chunk[0]}" --eval-modes "$rest" &
    pids+=($!)
    i=$((i + 1))
  done
}
SPECS=("$@")
lease wt "${HD_WT:-5}" --ppl-sets wikitext2
lease fw0 "${HD_FW:-2}" --ppl-sets fineweb_val --val-tokens 524288
lease fw1 "${HD_FW:-2}" --ppl-sets fineweb_val --val-tokens 524288 --val-skip 256
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
