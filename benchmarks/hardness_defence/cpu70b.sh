#!/usr/bin/env bash
# One Llama-3.1-70B quality arm on a CPU node (for when no GPU node is up): the weights fetched once per node into node
# RAM (fetch.sh, under a lock), then qat.py with --device-map cpu on $HD_THREADS threads (default 96, one a core), its
# outputs in $OUT/NAME and its log in $OUT/NAME.log. The CPU's noise draws and bf16 sums aren't the GPU's, so a CPU arm
# is compared with a CPU bf16 arm. Remove the weights when the node's arms are done (fetch.sh).
#   bash benchmarks/hardness_defence/cpu70b.sh NAME QAT_ARGS...
set -euo pipefail
name=$1
shift
here=$(cd "$(dirname "$0")" && pwd)
dir=${HD_70B:-/dev/shm/compute-research-hd/Meta-Llama-3.1-70B}
mkdir -p "$(dirname "$dir")"
( flock 9
  [ -f "$dir/.fetched" ] || { bash "$here/fetch.sh" unsloth/Meta-Llama-3.1-70B "$dir" && touch "$dir/.fetched"; }
) 9>"$(dirname "$dir")/fetch.lock"
export OMP_NUM_THREADS=${HD_THREADS:-96}
bash "$here/../hardness_useful/run_arm.sh" "$name" --model "$dir" --device-map cpu --tokens 0 --tasks "" "$@" \
  > "${OUT:?}/$name.log" 2>&1
