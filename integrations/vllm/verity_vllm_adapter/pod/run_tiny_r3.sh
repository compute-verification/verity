#!/bin/bash
# Pod-side: the R3 row at bundle scale (§16.8) -- SmolLM2-135M (case B2), 3 requests (source, miss, hit of
# 2 blocks), 4 tokens each, prefix caching ON in production and in the replay (cross-request KV rule),
# one fresh replay with the cache off (values stored), verify, then `verity_vllm.bundle pack` (weights omitted from
# the tarball, referenced by checkpoint sha). Runs in the GPU-idle window of the B1 queue (its verify stage is CPU-only):
# waits until no production/replay process is running and a verify process is.
# usage: nohup run_tiny_r3.sh <REV> > /workspace/logs/r3-tiny.log 2>&1 &
set -u
REV=$1
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VERITY_WORKER=W10
PY=/workspace/venv/bin/python
ulimit -n 65536
CASE=B2; TAG=r3-tiny
R=/workspace/results/$CASE/$TAG; mkdir -p $R /workspace/logs
L=/workspace/logs/$CASE-$TAG
CK=manifests/checkpoints.json
WHEEL=7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a $R/pipeline.log; }
TOK=$($PY -c "import json;print([c for c in json.load(open('$CK'))['checkpoints'] if c['repo']=='HuggingFaceTB/SmolLM2-135M'][0]['local_path'])")
$PY -m verity_vllm_adapter.state_rows tiny-workload --tokenizer $TOK --model HuggingFaceTB/SmolLM2-135M --out $R/workload.json > $L-workload.log 2>&1 || { log "workload FAILED"; exit 1; }
log "workload: $(cat $L-workload.log | tail -1 | cut -c1-300)"
gpu_window() {  # true when the B1 queue is in a CPU-only stage
  ! pgrep -f "verity_vllm_adapter.(production|replay) --case B1" > /dev/null && pgrep -f "verity_vllm_adapter.verify" > /dev/null
}
log "waiting for a GPU-idle window (B1 verify stage)"
while ! gpu_window; do sleep 20; done
log "window open; nvidia-smi used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
EA=(--engine-arg enable_prefix_caching=true --engine-arg gpu_memory_utilization=0.10 --engine-arg max_num_seqs=4)
$PY -m verity_vllm_adapter.production --case $CASE --checkpoints $CK --workload $R/workload.json --out $R/production --adapter-revision $REV \
  --wheel-sha256 $WHEEL --vllm-commit d9105ea80 --enforce-eager "${EA[@]}" > $L-production.log 2>&1 \
  || { log "production FAILED: $(grep -E 'CONTAMINATED|Error|error' $L-production.log | tail -1 | cut -c1-300)"; exit 1; }
log "production done: $(grep profile_id= $L-production.log | tail -1 | cut -c1-300) cached=$($PY -c "import json;w=json.load(open('$R/production/workload_exercised.json'));print(w['num_cached_tokens_per_request'])")"
for MODE in ref1 ref2 commit fresh; do
  case $MODE in
    commit) M="--mode commit --weights-values store --prefix-caching";;   # the adapter verifier opens weight leaves from values.bin; the tarball omits them
    fresh)  M="--mode commit --weights-values hash";;   # cache off, values stored: byte comparison of B-fresh vs A at the cached positions
    *)      M="--mode reference --prefix-caching";;
  esac
  rm -rf $R/$MODE
  $PY -m verity_vllm_adapter.replay --case $CASE --checkpoints $CK --production $R/production --out $R/$MODE $M --adapter-revision $REV --arena-gb 1 > $L-$MODE.log 2>&1 \
    || { log "$MODE FAILED: $(grep -E 'Error|error' $L-$MODE.log | tail -1 | cut -c1-300)"; exit 1; }
  log "$MODE done: $(grep DONE $L-$MODE.log | tail -1)"
done
$PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/ref1 --ref $R/ref2 --out $R/compare.json > $L-compare.log 2>&1
log "compare: $(head -1 $L-compare.log)"
$PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/fresh --out $R/compare_fresh.json > $L-compare-fresh.log 2>&1
log "compare cached-vs-fresh: $(sed -n 2p $L-compare-fresh.log)"
$PY -m verity_vllm_adapter.verify $R/commit --seed 7 --sample-size 1000 --negative-limit 4 --negatives N01,N02,N03,N04,N05,N06,N07,N13,N14 \
  --production $R/production --noninterference $R/compare.json --adapter-revision $REV --case $CASE --out $R/verify > $L-verify.log 2>&1
log "verify: $(grep RESULT $L-verify.log | tail -1)"
$PY -m verity_vllm_adapter.state_rows recompute-check $R/commit --out $R/recompute_check.json >> $R/pipeline.log 2>&1
$PY -m verity_vllm_adapter.state_rows cached-check $R/commit --out $R/cached_check.json >> $R/pipeline.log 2>&1
$PY -m verity_vllm_adapter.state_rows cached-vs-fresh $R/commit $R/fresh --out $R/cached_vs_fresh.json >> $R/pipeline.log 2>&1
EXP=$($PY -c "import json;print(json.load(open('$R/verify/verify.json'))['experiment_id'])")
$PY -m verity_vllm.bundle pack $R/commit --out $R/$EXP.tar.gz --production $R/production/production.jsonl --verify-dir $R/verify --name $EXP --weights omit > $L-pack.log 2>&1 \
  && log "bundle: $(ls -la $R/$EXP.tar.gz | awk '{print $5}') bytes $R/$EXP.tar.gz" || log "pack FAILED: $(tail -1 $L-pack.log | cut -c1-300)"
log "PIPELINE_DONE $CASE/$TAG"
