#!/bin/bash
# W1d pod-side pipeline for one case x tag (compiled-preserving replay, CONTRACT §16.1 rung (b)):
#   [production (stock compiled, existing recorder)] -> ref1 -> ref2 (hash-only, pass on) -> commit -> compare -> verify
# usage: pod_run.sh <CASE> <TAG> <REV> <PRODUCTION_DIR|new> [--workload W] [--seed N] [--sample-size N] [--negative-limit N]
#                   [--production-inductor-cache DIR] [--weights-values store|hash] [--arena-gb N]
#                   [--replay-extra "--cudagraph production --record-impl device"]   (rung (c))
# Every replay compile gets private, fresh TORCHINDUCTOR_CACHE_DIR / VLLM_CACHE_ROOT (the pass runs at compile time).
set -u
CASE=$1; TAG=$2; REV=$3; PROD=$4; shift 4
WL=""; SEED=7; SS=1000; NL=4; PIC=""; WV=store; ARENA=8; RX=""
while [ $# -gt 0 ]; do case "$1" in
  --workload) WL=$2; shift;; --seed) SEED=$2; shift;; --sample-size) SS=$2; shift;; --negative-limit) NL=$2; shift;;
  --production-inductor-cache) PIC=$2; shift;; --weights-values) WV=$2; shift;; --arena-gb) ARENA=$2; shift;; --replay-extra) RX=$2; shift;; esac; shift; done
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
PY=${PY:-/workspace/venv/bin/python}
R=/workspace/results/$CASE/$TAG; mkdir -p $R /workspace/logs
CK=manifests/checkpoints.json
WHEEL=0198513ebd5561f154d9a5a241cdaacd1a7a211bff66aa8997e9e62481eb9120
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a $R/pipeline.log; }
log "start $CASE/$TAG rev=$REV production=$PROD"
if [ "$PROD" = new ]; then
  PROD=$R/production; PIC=/workspace/inductor_cache/production-$TAG
  rm -rf $PIC /workspace/vllm_cache/production-$TAG
  # stock compiled production; only the cache *location* is set (recorded in the row) so the kernel sources can be compared
  TORCHINDUCTOR_CACHE_DIR=$PIC VLLM_CACHE_ROOT=/workspace/vllm_cache/production-$TAG \
  $PY -m verity_vllm_adapter.production --case $CASE --checkpoints $CK --workload ${WL:-/workspace/results/$CASE/workload.json} --out $PROD --adapter-revision $REV --wheel-sha256 $WHEEL --vllm-commit d9105ea80 > /workspace/logs/$CASE-$TAG-production.log 2>&1 || { log "production FAILED: $(grep -E 'CONTAMINATED|Error|error' /workspace/logs/$CASE-$TAG-production.log | tail -1 | cut -c1-300)"; exit 1; }
  log "production done: $(grep profile_id= /workspace/logs/$CASE-$TAG-production.log | tail -1)"
fi
[ -n "$PIC" ] || { log "need --production-inductor-cache for an existing production"; exit 1; }
for MODE in ref1 ref2 commit; do
  if [ $MODE = commit ]; then M="--mode commit --weights-values $WV"; else M="--mode reference"; fi
  rm -rf $R/$MODE
  TORCHINDUCTOR_CACHE_DIR=/workspace/inductor_cache/$TAG-$MODE VLLM_CACHE_ROOT=/workspace/vllm_cache/$TAG-$MODE \
  $PY -m verity_vllm_adapter.compiled_fx.replay --case $CASE --checkpoints $CK --production $PROD --out $R/$MODE $M --adapter-revision $REV --arena-gb $ARENA --production-inductor-cache $PIC $RX > /workspace/logs/$CASE-$TAG-$MODE.log 2>&1 || { log "$MODE FAILED: $(grep -v '^\[rank0\]:\s*$' /workspace/logs/$CASE-$TAG-$MODE.log | grep -E 'Error|error' | tail -1 | cut -c1-300)"; exit 1; }
  log "$MODE done: $(grep -E '^run |kernel identity|description:|reference sidecars' /workspace/logs/$CASE-$TAG-$MODE.log | tr '\n' ' ' | cut -c1-400)"
done
$PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/ref1 --ref $R/ref2 --out $R/compare.json > /workspace/logs/$CASE-$TAG-compare.log 2>&1
log "compare: $(head -1 /workspace/logs/$CASE-$TAG-compare.log)"
WR=""; [ "$WV" = hash ] && WR="--weights-resolver checkpoint --checkpoints $CK"   # weights by checkpoint sha (§16.8): re-derived through vLLM's loader, re-hashed against the index
$PY -m verity_vllm_adapter.verify $R/commit --seed $SEED --sample-size $SS --negative-limit $NL --production $PROD --noninterference $R/compare.json --adapter-revision $REV --case $CASE --worker W1d $WR --out $R/verify > /workspace/logs/$CASE-$TAG-verify.log 2>&1
log "verify: $(grep RESULT /workspace/logs/$CASE-$TAG-verify.log | tail -1 | cut -c1-400)"
log "PIPELINE_DONE $CASE/$TAG"
