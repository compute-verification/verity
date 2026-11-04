#!/usr/bin/env bash
# One quality arm on node 2, under its GPU lease: NGPU GPUs for at most 30 minutes, preemptible (fill work), qat.py's
# outputs in $OUT/NAME and its log in $OUT/NAME.log.
#   bash benchmarks/hardness_defence/arm.sh NAME NGPU QAT_ARGS...
set -euo pipefail
name=$1 ngpu=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
GPU_LEASE_WHO=${GPU_LEASE_WHO:-compute-research-hd} gpu-lease "$ngpu" --wait --max-min 30 --preemptible -- \
  bash "$here/../hardness_useful/run_arm.sh" "$name" "$@" > "${OUT:?}/$name.log" 2>&1
