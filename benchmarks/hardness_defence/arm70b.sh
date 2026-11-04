#!/usr/bin/env bash
# One Llama-3.1-70B quality arm as three leases of 2 GPUs, each under arm.sh's 30 minutes: WikiText-2's test split, and
# FineWeb-Edu val's first and second 524,288 tokens, which together are qat.py's default 1,048,576 (their nll summed over
# their tokens). The weights are read from node RAM (fetch.sh).
#   bash benchmarks/hardness_defence/arm70b.sh NAME QAT_ARGS...
set -euo pipefail
name=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
common=(--model "${HD_70B:-/dev/shm/compute-research-hd/Meta-Llama-3.1-70B}" --device-map auto --tokens 0 --tasks ""
  --eval-modes "")
bash "$here/arm.sh" "$name-wt" 2 "${common[@]}" --ppl-sets wikitext2 "$@" &
p1=$!
bash "$here/arm.sh" "$name-fw0" 2 "${common[@]}" --ppl-sets fineweb_val --val-tokens 524288 "$@" &
p2=$!
bash "$here/arm.sh" "$name-fw1" 2 "${common[@]}" --ppl-sets fineweb_val --val-tokens 524288 --val-skip 256 "$@" &
p3=$!
rc=0
for p in $p1 $p2 $p3; do wait "$p" || rc=1; done
exit $rc
