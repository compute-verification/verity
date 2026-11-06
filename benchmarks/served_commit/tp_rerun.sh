#!/usr/bin/env bash
# tp_rerun.py (a served window replayed in fresh engine processes, compared bit for bit) on one 8-GPU H200 node, from the source
# tree's root:
#   RESEARCH_RUN_DIR=DIR bash benchmarks/served_commit/tp_rerun.sh
# `run` in RUNS (default 3) engine processes one after another, then `compare` of each with the first. ARMS (default
# served,served-2; `all` for every arm of tp_replay); ROW, CASE, TP, LONG, GPU_UTIL, MAX_MODEL_LEN, MAX_BATCHED and VENV as in
# tp_replay.sh. The fingerprints (about 100 MB a rank and arm for GLM-5.3's window) go to DUMP, by default a new directory under
# /workspace outside the run dir, removed after the comparison unless KEEP_DUMP=1.
# Out: $RESEARCH_RUN_DIR/out/rerun/run<i> (versions.json, arms.jsonl, run.json) and cross.json.
set -euo pipefail
ROW=${ROW:-glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager}
# shellcheck source=tp_env.sh
source "$PWD/benchmarks/served_commit/tp_env.sh"
OUT=$RESEARCH_RUN_DIR/out/rerun
DUMP=${DUMP:-$(mktemp -d /workspace/tp-rerun.XXXXXX)}
for i in $(seq 0 $((${RUNS:-3} - 1))); do
  python -u -m served_commit.tp_rerun run --workload "workloads/$ROW.json" --case "${CASE:-GLM53}" --tp "${TP:-8}" \
    --arms "${ARMS:-served,served-2}" --long "${LONG:-3072}" --engine-arg "gpu_memory_utilization=${GPU_UTIL:-0.90}" \
    --engine-arg "max_model_len=${MAX_MODEL_LEN:-4096}" --engine-arg "max_num_batched_tokens=${MAX_BATCHED:-2048}" \
    --engine-arg disable_custom_all_reduce=false --out "$OUT/run$i" --dump "$DUMP/run$i"
done
python -u -m served_commit.tp_rerun compare "$OUT"/run* --out "$OUT/cross.json"
[ "${KEEP_DUMP:-0}" = 1 ] || rm -rf "$DUMP"
