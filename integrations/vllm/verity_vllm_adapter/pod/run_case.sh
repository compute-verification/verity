#!/bin/bash
# Pod-side pipeline for one case × profile: [production] -> ref1 -> ref2 -> commit -> compare -> verify.
# usage: run_case.sh <CASE> <TAG> <REV> [--production] [--enforce-eager] [--custom-ops all|none] [--production-only]
#                    [--compare-tokens <production.jsonl>] [--weights-values store|hash|checkpoint] [--arena-gb N]
#                    [--replay-exec eager|compiled-nograph] [--workload <workload.json>] [--limit N]
#                    [--challenge security|diagnostic] [--seed N] [--q FRAC] [--s FRAC] [--verify-only] [--verify-out DIR] [--worker ID]
#                    [--max-num-batched-tokens N]   (with --production: a declared scheduler flag of the production profile, M2 varied
#                                                    batch composition, replay follows the manifest; without it: the replay-only cap
#                                                    for host-memory-bound pods, W3c, recorded as a replay restriction)
#                    [--order reversed|interleaved] [--extra-ref <run dir>]   (M2: varied batch composition; --extra-ref = another
#                                                    profile's run for the cross-profile value comparison, diagnostic only)
#                    [--tensor-parallel-size N] [--rank-tag R]   (lane TP / profile ports: N is a declared production flag (manifest
#                                                    parallelism), R tags this replay's value ids with s=<rank> and writes the
#                                                    replay dirs as <mode>/rank<R>; conformance/pod/run_conformance.sh drives both)
# env: PY (python of the pinned venv), WHEEL (sha256 of the installed vLLM wheel), VLLM_COMMIT, VERITY_WORKER.
# Every stage runs with its own vLLM / Inductor / Triton cache root under /workspace/cache/<CASE>-<TAG>-<stage>
# so the generated-kernel sources of production and of each replay are collectable and comparable (§16.1).
# The challenge defaults to security mode (the verifier's own secret randomness, GUIDANCE C1); --seed is only
# meaningful with --challenge diagnostic and is refused otherwise.  --verify-only re-runs the verifier on an
# existing $R/commit (the replay/commit/compare stages are skipped), e.g. the second row over the same capture.
set -u
CASE=$1; TAG=$2; REV=$3; shift 3
DO_PROD=0; PROD_ONLY=0; EAGER=""; CUSTOM=""; CMP=""; WV="store"; ARENA=24; SEED=""; EXEC="eager"; WL=""; LIMIT=""
CHALLENGE=security; Q=1; S=1/4096; VERIFY_ONLY=0; VOUT=""; WORKER=${VERITY_WORKER:-}; MNBT=""; ORDER=""; XREF=""; SP=""; TPARG=""; RANK=""; RSUB=""; RDOT=""
QUANT=""   # [QZ, additive] --quantization <scheme>: declared production flag (port b1-fp8); replay reads it from the manifest
while [ $# -gt 0 ]; do case "$1" in
  --quantization) QUANT="--quantization $2"; shift;;
  --tp) TPARG="--tensor-parallel-size $2"; shift;;   # lane TP alias of --tensor-parallel-size (declared production flag; replay follows the manifest)
  --production) DO_PROD=1;; --production-only) DO_PROD=1; PROD_ONLY=1;; --enforce-eager) EAGER="--enforce-eager";;
  --custom-ops) CUSTOM="--custom-ops $2"; shift;; --compare-tokens) CMP="--compare-tokens $2"; shift;; --weights-values) WV=$2; shift;;
  --arena-gb) ARENA=$2; shift;; --seed) SEED=$2; shift;; --replay-exec) EXEC=$2; shift;; --workload) WL=$2; shift;; --limit) LIMIT="--limit $2"; shift;;
  --challenge) CHALLENGE=$2; shift;; --q) Q=$2; shift;; --s) S=$2; shift;;
  --verify-only) VERIFY_ONLY=1;; --verify-out) VOUT=$2; shift;; --worker) WORKER=$2; shift;;
  --max-num-batched-tokens) MNBT="--max-num-batched-tokens $2"; shift;;
  --order) ORDER="--order $2"; shift;; --extra-ref) XREF="$XREF --ref $2"; shift;;
  --semantic-profile) SP="--semantic-profile $2"; shift;;   # RC: the verifier's semantic profile (Qwen2 for B1)
  --tensor-parallel-size) TPARG="--tensor-parallel-size $2"; shift;; --rank-tag) RANK="--rank-tag $2"; RSUB="/rank$2"; RDOT=".rank$2"; shift;; esac; shift; done
if [ "$CHALLENGE" = security ] && [ -n "$SEED" ]; then echo "--seed is a diagnostic-mode argument; security mode uses the verifier's own randomness"; exit 2; fi
if [ "$CHALLENGE" = diagnostic ] && [ -z "$SEED" ]; then SEED=7; fi
SEEDARG=""; [ -n "$SEED" ] && SEEDARG="--seed $SEED"
WORKERARG=""; [ -n "$WORKER" ] && WORKERARG="--worker $WORKER"
# --max-num-batched-tokens routing (see usage): declared on production when production runs here (replay.py then
# takes it from the manifest), otherwise the W3c replay-only override.
PROD_MNBT=""; REPLAY_MNBT="$MNBT"; [ $DO_PROD = 1 ] && { PROD_MNBT="$MNBT"; REPLAY_MNBT=""; }
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
PY=${PY:-/workspace/venv/bin/python}   # env override for pods whose pinned venv lives elsewhere (A100: /workspace/venv-nightly; 4090 cu129: /workspace/venv-cu129)
R=/workspace/results/$CASE/$TAG; mkdir -p $R /workspace/logs /workspace/cache
echo "[$(date -u +%H:%M:%SZ)] python=$PY vllm=$($PY -c 'import vllm; print(vllm.__version__)' 2>/dev/null)" | tee -a $R/pipeline.log
CK=manifests/checkpoints.json
WHEEL=${WHEEL:-0198513ebd5561f154d9a5a241cdaacd1a7a211bff66aa8997e9e62481eb9120}   # env override: the cu129 build of the same commit (W10 pod, driver 570) is 7aa52ac7...
VLLM_COMMIT=${VLLM_COMMIT:-d9105ea80}
WL=${WL:-/workspace/results/$CASE/workload.json}
cache_env() {  # $1 = stage
  export VLLM_CACHE_ROOT=/workspace/cache/$CASE-$TAG-$1
  export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton
  mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
}
echo "[$(date -u +%H:%M:%SZ)] start $CASE/$TAG rev=$REV eager=$EAGER custom_ops=$CUSTOM replay_exec=$EXEC challenge=$CHALLENGE workload=$WL" | tee -a $R/pipeline.log
if [ $DO_PROD = 1 ]; then
  [ -f $WL ] || echo "missing workload $WL"
  cache_env production
  $PY -m verity_vllm_adapter.production --case $CASE --checkpoints $CK --workload $WL --out $R/production --adapter-revision $REV --wheel-sha256 $WHEEL --vllm-commit $VLLM_COMMIT $EAGER $CUSTOM $CMP $ORDER $PROD_MNBT $TPARG $QUANT > /workspace/logs/$CASE-$TAG-production.log 2>&1 || { echo "production FAILED: $(grep -E 'CONTAMINATED|Error|error' /workspace/logs/$CASE-$TAG-production.log | tail -1 | cut -c1-300)" | tee -a $R/pipeline.log; exit 1; }
  echo "[$(date -u +%H:%M:%SZ)] production done: $(grep profile_id= /workspace/logs/$CASE-$TAG-production.log | tail -1)" | tee -a $R/pipeline.log
  [ $PROD_ONLY = 1 ] && { echo "[$(date -u +%H:%M:%SZ)] PRODUCTION_ONLY_DONE $CASE/$TAG" | tee -a $R/pipeline.log; exit 0; }
fi
if [ $VERIFY_ONLY = 0 ]; then
for MODE in ref1 ref2 commit; do
  if [ $MODE = commit ]; then M="--mode commit --weights-values $WV"; else M="--mode reference"; fi
  rm -rf $R/$MODE$RSUB
  cache_env $MODE
  $PY -m verity_vllm_adapter.replay --case $CASE --checkpoints $CK --production $R/production --out $R/$MODE$RSUB $M --adapter-revision $REV --arena-gb $ARENA --replay-exec $EXEC $LIMIT $REPLAY_MNBT $RANK > /workspace/logs/$CASE-$TAG-$MODE.log 2>&1 || { echo "$MODE FAILED" | tee -a $R/pipeline.log; exit 1; }
  echo "[$(date -u +%H:%M:%SZ)] $MODE done: $(grep DONE /workspace/logs/$CASE-$TAG-$MODE.log | tail -1)" | tee -a $R/pipeline.log
done
$PY -m verity_vllm_adapter.compare --commit $R/commit$RSUB --ref $R/ref1$RSUB --ref $R/ref2$RSUB --out $R/compare$RDOT.json > /workspace/logs/$CASE-$TAG-compare.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] compare: $(head -1 /workspace/logs/$CASE-$TAG-compare.log)" | tee -a $R/pipeline.log
if [ -n "$XREF" ]; then
  # cross-profile comparison (M2 vs M1: same requests under another batch composition), diagnostic only
  $PY -m verity_vllm_adapter.compare --commit $R/commit$RSUB $XREF --ignore-profile --out $R/compare_cross$RDOT.json > /workspace/logs/$CASE-$TAG-compare-cross.log 2>&1
  echo "[$(date -u +%H:%M:%SZ)] compare-cross: $(head -1 /workspace/logs/$CASE-$TAG-compare-cross.log)" | tee -a $R/pipeline.log
fi
fi
cache_env verify  # W1cF: the verifier compiles the bundled kernels itself (F10 PTX identity, F8 autotune pinning); never the replay's caches
WFM=""; [ "$WV" = checkpoint ] && WFM="--weights-from-model $($PY -c "import json,sys; c=json.load(open('$CK')); import verity_vllm_adapter.profile as p; print(p.checkpoint_for('$CASE','$CK')['local_path'])")"
VOUT=${VOUT:-$R/verify-$CHALLENGE$RSUB}
$PY -m verity_vllm_adapter.verify $R/commit$RSUB --challenge $CHALLENGE $SEEDARG --q $Q --s $S --coverage-size ${COVERAGE:-1000} --negative-limit 4 ${VERIFY_EXTRA:-} --production $R/production --checkpoints $CK --noninterference $R/compare$RDOT.json --adapter-revision $REV --case $CASE $WFM $WORKERARG $SP --out $VOUT > /workspace/logs/$CASE-$TAG-verify-$CHALLENGE.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] verify ($CHALLENGE -> $VOUT): $(grep RESULT /workspace/logs/$CASE-$TAG-verify-$CHALLENGE.log | tail -1)" | tee -a $R/pipeline.log
echo "[$(date -u +%H:%M:%SZ)] PIPELINE_DONE $CASE/$TAG" | tee -a $R/pipeline.log
