#!/bin/bash
# [COMP, CX1] pod-side pipeline for the held-out cell b1-tp2-compiled (compiled execution x tensor_parallel_size=2 on B1):
#   production (stock compiled, TP=2, two independent processes) -> ref1 -> ref2 -> commit (compiled-fx capture per rank,
#   tp_merge) -> compare.  Mirrors compiled_fx/pod_run.sh with the TP flag on production and the per-rank compiled-fx replay
#   (compiled_fx/replay.py run_tp).  Correspondence is run separately (verity_vllm.correspondence.tp2 --port .../b1-tp2-compiled).
# usage: run_cx1.sh <TAG> <REV> [--workload W] [--stage production|replay|all] [--weights-values store|hash] [--arena-gb N]
#                   [--modes "ref1 ref2 commit"] [--skip-second-production]
set -u
TAG=$1; REV=$2; shift 2
WL=/workspace/veritor/e2e/workload-B1-512.json; STAGE=all; WV=store; ARENA=8; MODES="ref1 ref2 commit"; SECOND=1
while [ $# -gt 0 ]; do case "$1" in
  --workload) WL=$2; shift;; --stage) STAGE=$2; shift;; --weights-values) WV=$2; shift;; --arena-gb) ARENA=$2; shift;;
  --modes) MODES=$2; shift;; --skip-second-production) SECOND=0;; *) echo "unknown argument $1"; exit 2;; esac; shift; done
CASE=B1
cd /workspace/veritor/vllm-poc || exit 1
export PYTHONPATH=.:..:../src HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=1
PY=${PY:-/workspace/venv/bin/python}
R=/workspace/results/comp/$TAG; mkdir -p $R /workspace/logs /workspace/cache
CK=manifests/checkpoints.json
WHEEL=0198513ebd5561f154d9a5a241cdaacd1a7a211bff66aa8997e9e62481eb9120
log() { echo "[$(date -u +%H:%M:%SZ)] cx1/$TAG: $*" | tee -a $R/pipeline.log; }
PIC=/workspace/cache/$TAG-production   # both the Inductor cache and vLLM's cache root of the production live under it (per-rank dirs rank_<r>_0)

production() {
  local sub=$1 out=$2 cache=$3
  rm -rf $cache; mkdir -p $cache
  log "production $sub start (compiled, tensor_parallel_size=2, workload $WL)"
  TORCHINDUCTOR_CACHE_DIR=$cache/inductor VLLM_CACHE_ROOT=$cache/vllm \
  $PY -m verity_vllm_adapter.production --case $CASE --checkpoints $CK --workload $WL --out $out --adapter-revision $REV --wheel-sha256 $WHEEL --vllm-commit d9105ea80 \
      --tensor-parallel-size 2 > /workspace/logs/cx1-$TAG-$sub.log 2>&1 || { log "production $sub FAILED: $(grep -E 'CONTAMINATED|Error|error' /workspace/logs/cx1-$TAG-$sub.log | tail -1 | cut -c1-300)"; exit 1; }
  log "production $sub done: $(grep profile_id= /workspace/logs/cx1-$TAG-$sub.log | tail -1)"
}

replay() {
  [ -d $R/production ] || { log "no $R/production"; exit 1; }
  for MODE in $MODES; do
    if [ $MODE = commit ]; then M="--mode commit --weights-values $WV"; else M="--mode reference"; fi
    rm -rf $R/$MODE /workspace/cache/$TAG-$MODE
    log "$MODE start"
    TORCHINDUCTOR_CACHE_DIR=/workspace/cache/$TAG-$MODE/inductor VLLM_CACHE_ROOT=/workspace/cache/$TAG-$MODE/vllm \
    $PY -m verity_vllm_adapter.compiled_fx.replay --case $CASE --checkpoints $CK --production $R/production --out $R/$MODE $M --adapter-revision $REV \
        --arena-gb $ARENA --production-inductor-cache $PIC > /workspace/logs/cx1-$TAG-$MODE.log 2>&1 || { log "$MODE FAILED: $(grep -v '^\[rank0\]:\s*$' /workspace/logs/cx1-$TAG-$MODE.log | grep -E 'Error|error|SystemExit' | tail -2 | cut -c1-400)"; exit 1; }
    log "$MODE done: $(grep -E '^run |^followed|kernel identity|per rank:|merge' /workspace/logs/cx1-$TAG-$MODE.log | tr '\n' ' ' | cut -c1-600)"
  done
  if [ -d $R/commit ] && [ -d $R/ref1 ] && [ -d $R/ref2 ]; then
    $PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/ref1 --ref $R/ref2 --out $R/compare.json > /workspace/logs/cx1-$TAG-compare.log 2>&1
    log "compare: $(head -1 /workspace/logs/cx1-$TAG-compare.log)"
  fi
}

log "start rev=$REV stage=$STAGE"
case "$STAGE" in
  production) production production $R/production $PIC; [ $SECOND = 1 ] && production production-r2 $R/production-r2 /workspace/cache/$TAG-production-r2;;
  replay) replay;;
  all) production production $R/production $PIC; [ $SECOND = 1 ] && production production-r2 $R/production-r2 /workspace/cache/$TAG-production-r2; replay;;
esac
log "PIPELINE_DONE $TAG"
