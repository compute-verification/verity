#!/usr/bin/env bash
# One arm of tp_decode.py (stock against the arm in one TP engine, eager), from the source tree's root on an 8-GPU H200 node:
#   ARM=rows RESEARCH_RUN_DIR=DIR bash benchmarks/served_commit/tp_decode.sh
# ARM: rows | rows-4k | rows-layer | rows-layer-4k.  RETAIN (default none): the committer's value retention, VERITY_RETAIN
# (none | host | device), which summary.json records as the committer took it.  BATCHES (default 1,8,32), PAIRS (default 3), ROW
# (the workload, default GLM-5.3's TP8 eager B1, whose request every batch rotates), CASE (default GLM53), TP (default 8), VENV (default
# /workspace/cp/venv312; the node's setup is tp_env.sh).  The engine: gpu_memory_utilization GPU_UTIL (0.92), max_model_len
# MAX_MODEL_LEN (288), max_num_batched_tokens MAX_BATCHED (2048, the workload's).
# Out: $RESEARCH_RUN_DIR/out/$ARM (runs.jsonl, summary.json, committer_make.json, versions.json, b<B>/ the warm-ups).
set -euo pipefail
ARM=${ARM:?ARM: rows | rows-4k | rows-layer | rows-layer-4k}
ROW=${ROW:-glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager}
# shellcheck source=tp_env.sh
source "$PWD/benchmarks/served_commit/tp_env.sh"
export NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
export VERITY_WINDOW_MB=256 VERITY_WINDOW_SLOTS=8 VERITY_RETAIN=${RETAIN:-none} VERITY_WEIGHTS_HASH=device
python -c "from verity_vllm.commit.committer import native_collect as N; N.ext()"
python -u -m served_commit.tp_decode --workload "workloads/$ROW.json" --case "${CASE:-GLM53}" --tp "${TP:-8}" --arm "$ARM" \
  --batches "${BATCHES:-1,8,32}" --pairs "${PAIRS:-3}" --engine-arg "gpu_memory_utilization=${GPU_UTIL:-0.92}" \
  --engine-arg "max_model_len=${MAX_MODEL_LEN:-288}" --engine-arg "max_num_batched_tokens=${MAX_BATCHED:-2048}" \
  --engine-arg disable_custom_all_reduce=false --out "$RESEARCH_RUN_DIR/out/$ARM"
