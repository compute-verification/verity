#!/bin/bash
# Pod-side pipeline for one state row (W10: R2 chunked prefill, R3 prefix-cache hits, R4 preemption):
#   production (declared --engine-arg overrides) -> ref1 -> ref2 -> commit -> compare -> verify -> row checks
# usage: run_state_row.sh <CASE> <TAG> <REV> <WORKLOAD.json> [--engine-arg K=V ...] [--replay-prefix-caching]
#                         [--replay-engine-arg K=V ...] [--fresh-ref] [--arena-gb N] [--seed N] [--production-only] [--skip-production]
# The replay engine re-applies production's overrides (state_rows.replay_engine_overrides); with
# --replay-prefix-caching every replay runs WITH the prefix cache under the cross-request KV rule and
# --fresh-ref adds one more reference replay with the cache off (cached-vs-fresh leaf comparison).
set -u
CASE=$1; TAG=$2; REV=$3; WL=$4; shift 4
EA=(); REA=(); RPC=""; FRESH=0; ARENA=24; SEED=7; PROD_ONLY=0; SKIP_PROD=0; QUICK=0; RESUME=0
while [ $# -gt 0 ]; do case "$1" in
  --engine-arg) EA+=(--engine-arg "$2"); shift;; --replay-engine-arg) REA+=(--engine-arg "$2"); shift;; --replay-prefix-caching) RPC="--prefix-caching";; --fresh-ref) FRESH=1;;
  --arena-gb) ARENA=$2; shift;; --seed) SEED=$2; shift;; --production-only) PROD_ONLY=1;; --skip-production) SKIP_PROD=1;; --quick) QUICK=1;; --resume-at-commit) SKIP_PROD=1; RESUME=1;; esac; shift; done
# --resume-at-commit: production and ref1/ref2 kept, rerun commit [fresh] -> compare -> verify (a commit that died).
ulimit -n 65536   # the commitment builder maps every replay unit's values.bin (R3: 965 units > the 1024 default)
# --quick: production + ref1 only (the R2 witness, the token/logprob record and the kv_unresolved check
# without the 60-min commit/verify tail); a later --skip-production run completes the row.
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VERITY_WORKER=W10
PY=${PY:-/workspace/venv/bin/python}
R=/workspace/results/$CASE/$TAG; mkdir -p $R /workspace/logs
L=/workspace/logs/$CASE-$TAG
CK=manifests/checkpoints.json
WHEEL=${WHEEL:-7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3}   # cu129 build of d9105ea80 (W10 pod, driver 570)
echo "[$(date -u +%H:%M:%SZ)] start $CASE/$TAG rev=$REV engine_args=[${EA[*]:-}] replay=$RPC replay_args=[${REA[*]:-}] fresh_ref=$FRESH" | tee -a $R/pipeline.log
if [ $SKIP_PROD = 0 ]; then
$PY -m verity_vllm_adapter.production --case $CASE --checkpoints $CK --workload $WL --out $R/production --adapter-revision $REV \
  --wheel-sha256 $WHEEL --vllm-commit d9105ea80 --enforce-eager "${EA[@]}" > $L-production.log 2>&1 \
  || { echo "production FAILED: $(grep -E 'CONTAMINATED|Error|error' $L-production.log | tail -1 | cut -c1-300)" | tee -a $R/pipeline.log; exit 1; }
echo "[$(date -u +%H:%M:%SZ)] production done: $(grep profile_id= $L-production.log | tail -1) preemptions=$($PY -c "import json;print(json.load(open('$R/production/workload_exercised.json'))['preemptions'])") cached_req=$($PY -c "import json;print(json.load(open('$R/production/workload_exercised.json'))['requests_with_cached_tokens'])")" | tee -a $R/pipeline.log
fi
[ $PROD_ONLY = 1 ] && { echo "[$(date -u +%H:%M:%SZ)] PRODUCTION_ONLY_DONE $CASE/$TAG" | tee -a $R/pipeline.log; exit 0; }
MODES="ref1 ref2 commit"; [ $FRESH = 1 ] && MODES="ref1 ref2 commit fresh"; [ $QUICK = 1 ] && MODES="ref1"
[ $RESUME = 1 ] && { MODES="commit"; [ $FRESH = 1 ] && MODES="commit fresh"; }
for MODE in $MODES; do
  case $MODE in
    commit) M="--mode commit --weights-values store $RPC";;
    fresh)  M="--mode reference --store-values";;   # prefix cache off, values stored: byte comparison of a hit request's fresh leaves vs its source's committed ones
    *)      M="--mode reference $RPC";;
  esac
  rm -rf $R/$MODE
  $PY -m verity_vllm_adapter.replay --case $CASE --checkpoints $CK --production $R/production --out $R/$MODE $M "${REA[@]}" --adapter-revision $REV --arena-gb $ARENA > $L-$MODE.log 2>&1 \
    || { echo "$MODE FAILED: $(grep -E 'Error|error' $L-$MODE.log | tail -1 | cut -c1-300)" | tee -a $R/pipeline.log; exit 1; }
  echo "[$(date -u +%H:%M:%SZ)] $MODE done: $(grep DONE $L-$MODE.log | tail -1)" | tee -a $R/pipeline.log
done
[ $QUICK = 1 ] && { echo "[$(date -u +%H:%M:%SZ)] QUICK_DONE $CASE/$TAG" | tee -a $R/pipeline.log; exit 0; }
$PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/ref1 --ref $R/ref2 --out $R/compare.json > $L-compare.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] compare: $(head -1 $L-compare.log)" | tee -a $R/pipeline.log
if [ $FRESH = 1 ]; then
  $PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/fresh --out $R/compare_fresh.json > $L-compare-fresh.log 2>&1
  echo "[$(date -u +%H:%M:%SZ)] compare cached-vs-fresh: $(sed -n 2p $L-compare-fresh.log)" | tee -a $R/pipeline.log
fi
$PY -m verity_vllm_adapter.verify $R/commit --seed $SEED --sample-size 1000 --negative-limit 4 --negatives N01,N02,N03,N04,N05,N06,N07,N13,N14 \
  --production $R/production --noninterference $R/compare.json --adapter-revision $REV --case $CASE --out $R/verify > $L-verify.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] verify: $(grep RESULT $L-verify.log | tail -1)" | tee -a $R/pipeline.log
$PY -m verity_vllm_adapter.state_rows recompute-check $R/commit --out $R/recompute_check.json >> $R/pipeline.log 2>&1
$PY -m verity_vllm_adapter.state_rows cached-check $R/commit --out $R/cached_check.json >> $R/pipeline.log 2>&1
if [ $FRESH = 1 ]; then
  $PY -m verity_vllm_adapter.state_rows cached-vs-fresh $R/commit $R/fresh --out $R/cached_vs_fresh.json >> $R/pipeline.log 2>&1   # prints its scalar summary
fi
echo "[$(date -u +%H:%M:%SZ)] PIPELINE_DONE $CASE/$TAG" | tee -a $R/pipeline.log
