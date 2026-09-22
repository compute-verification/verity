#!/usr/bin/env bash
# CX2 structure smoke (lane COMP): compiled production of W9's MoE checkpoint on ONE A100, no capture.
# Scores the *structural* rows of composition-campaign.md §2 (the Inductor partition, `vllm.moe_forward` as an extern
# node, the fused sigmoid*mul pointwise kernel, the kernel-set) from the production's Inductor cache — C0 only.
#   bash run_cx2_smoke.sh <adapter-rev>
set -euo pipefail
REV=${1:?adapter revision}
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/veritor:/workspace/veritor/vllm-poc:/workspace/veritor/src
export VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_VISIBLE_DEVICES=0
PY=/workspace/venv/bin/python
R=/workspace/results/comp/cx2; mkdir -p $R /workspace/logs
CK=/workspace/veritor/manifests/checkpoints.json
WHEEL=0198513ebd5561f154d9a5a241cdaacd1a7a211bff66aa8997e9e62481eb9120
REPO=Qwen/Qwen1.5-MoE-A2.7B-Chat; REVISION=ec052fda178e241c7c443468d2fa1db6618996be
log() { echo "[$(date -u +%H:%M:%SZ)] cx2-smoke: $*" | tee -a $R/pipeline.log; }

log "download $REPO@$REVISION start"
SNAP=$($PY - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download("$REPO", revision="$REVISION", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab*"]))
EOF
)
log "download done: $SNAP"

log "workload start"
HF_HUB_OFFLINE=1 $PY -m verity_vllm_sampler.workload --model $SNAP --revision $REVISION --out $R/workload.json > /workspace/logs/cx2-workload.log 2>&1
log "workload done: $(python3 -c "import json;print(len(json.load(open('$R/workload.json'))['requests']))") requests"

CACHE=/workspace/cache/cx2-production; rm -rf $CACHE; mkdir -p $CACHE
log "production start (compiled, TP=1, one A100, max_num_batched_tokens=16384)"
HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor VLLM_CACHE_ROOT=$CACHE/vllm \
$PY -m verity_vllm_adapter.production --case M1 --checkpoints $CK --workload $R/workload.json --out $R/production \
    --adapter-revision $REV --wheel-sha256 $WHEEL --vllm-commit d9105ea80 --max-num-batched-tokens 16384 \
    > /workspace/logs/cx2-production.log 2>&1 || { log "production FAILED: $(grep -E 'Error|error' /workspace/logs/cx2-production.log | tail -1 | cut -c1-300)"; exit 1; }
log "production done: $(grep profile_id= /workspace/logs/cx2-production.log | tail -1)"

# the structural record: every Inductor wrapper (the partition and its extern nodes) + kernel bodies, no values
mkdir -p $R/inductor
# wrappers: Inductor emits `def call(args)` (B1, CX1) or the class form `def call(self, args)` (this cell: dynamic-range compile)
find $CACHE/inductor -name "*.py" -not -path "*/triton/*" | while read f; do
  if grep -qE "def call\((self, )?args\)" "$f"; then cp "$f" $R/inductor/wrapper-$(basename $(dirname $f))-$(basename $f); fi
done
find $CACHE/inductor -name "*.py" -path "*/triton/*" | while read f; do cp "$f" $R/inductor/kernel-$(basename $(dirname $f))-$(basename $f); done || true
cp $CACHE/vllm/torch_compile_cache/*/rank_0_0/backbone/computation_graph.py $R/inductor/ 2>/dev/null || true
log "inductor record: $(ls $R/inductor | grep -c wrapper) wrappers, $(ls $R/inductor | grep -c kernel) kernel files"
for w in $R/inductor/wrapper-*; do
  echo "-- $(basename $w): $(sed -n '/def call(\(self, \)\?args)/,$p' $w | grep -o '[a-zA-Z_0-9]*\.run(\|torch\.ops\.[a-zA-Z_0-9.]*(' | sed -E 's/_[0-9]+\.run\(/.run(/' | sort | uniq -c | sort -rn | tr -s ' ' | tr '\n' ';')" | tee -a $R/pipeline.log
done
log "SMOKE_DONE"
