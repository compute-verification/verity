#!/usr/bin/env bash
# tp_replay.py (a drawn request's replay: same batch, fillers, alone, another batch; cold and warm prefix) in one TP engine, eager,
# from the source tree's root on an 8-GPU H200 node:
#   RESEARCH_RUN_DIR=DIR bash benchmarks/served_commit/tp_replay.sh
# ROW (the workload, default GLM-5.3's TP8 eager B1, whose prompt every request tiles), CASE (default GLM53), TP (default 8), REPS
# (pass 2's timed runs of every arm, default 2), VENV (default /workspace/cp/venv312; the node's setup is tp_env.sh).  The engine:
# prefix caching on, gpu_memory_utilization GPU_UTIL (0.90), max_model_len MAX_MODEL_LEN (4096: the longest request is 3,336
# tokens), max_num_batched_tokens MAX_BATCHED (2048, the workload's).
# Out: $RESEARCH_RUN_DIR/out/replay (runs.jsonl, summary.json, versions.json).
set -euo pipefail
ROW=${ROW:-glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager}
# shellcheck source=tp_env.sh
source "$PWD/benchmarks/served_commit/tp_env.sh"
python -u -m served_commit.tp_replay --workload "workloads/$ROW.json" --case "${CASE:-GLM53}" --tp "${TP:-8}" --reps "${REPS:-2}" \
  --engine-arg "gpu_memory_utilization=${GPU_UTIL:-0.90}" --engine-arg "max_model_len=${MAX_MODEL_LEN:-4096}" \
  --engine-arg "max_num_batched_tokens=${MAX_BATCHED:-2048}" --engine-arg disable_custom_all_reduce=false \
  --out "$RESEARCH_RUN_DIR/out/replay"
