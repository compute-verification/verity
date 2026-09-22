#!/bin/bash
# RC: the *full-preparation control* of workstream C, experiment 1 -- every request of the production
# batch replayed, captured and committed (run_case.sh's ref1/ref2/commit/compare stages, the reference
# eager B1 profile), the weights rewritten to format v2 (one leaf per parameter row, as the selective
# bundles commit them) and the reference verifier run at the same q=1, s=1/4096, coverage 200 and no
# negatives as the selective verifier, so the check phase is comparable.  Times per stage go to
# $R/control_timing.json.
# usage: run_control.sh <REV> [--semantic-profile <json>] [--arena-gb N]
set -u
REV=$1; shift
SP=manifests/semantic-profiles/qwen2-eager-batch-invariant-v1.json; ARENA=48
while [ $# -gt 0 ]; do case "$1" in --semantic-profile) SP=$2; shift;; --arena-gb) ARENA=$2; shift;; esac; shift; done
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
PY=${PY:-/workspace/venv-cu129/bin/python}
CASE=B1; TAG=control; R=/workspace/results/$CASE/$TAG; CK=manifests/checkpoints.json
mkdir -p /workspace/logs
LOG=$R/pipeline.log
T=$R/control_timing.json; echo "{}" > $T
stamp() { $PY - "$T" "$1" "$2" <<'EOF'
import json, sys
p, k, v = sys.argv[1:4]
d = json.load(open(p)); d[k] = float(v); json.dump(d, open(p, "w"), indent=1)
EOF
}
cache_env() { export VLLM_CACHE_ROOT=/workspace/cache/$CASE-$TAG-$1; export TORCHINDUCTOR_CACHE_DIR=$VLLM_CACHE_ROOT/inductor TRITON_CACHE_DIR=$VLLM_CACHE_ROOT/triton; mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR; }
echo "[$(date -u +%H:%M:%SZ)] control start rev=$REV arena=$ARENA" | tee -a $LOG
for MODE in ref1 ref2 commit; do
  if [ $MODE = commit ]; then M="--mode commit --weights-values store"; else M="--mode reference"; fi
  rm -rf $R/$MODE; cache_env $MODE
  t0=$(date +%s.%N)
  $PY -m verity_vllm_adapter.replay --case $CASE --checkpoints $CK --production $R/production --out $R/$MODE $M --adapter-revision $REV --arena-gb $ARENA --replay-exec eager > /workspace/logs/$CASE-$TAG-$MODE.log 2>&1 || { echo "$MODE FAILED" | tee -a $LOG; exit 1; }
  stamp "${MODE}_wall_s" "$($PY -c "import time; print(time.time()-$t0)")"
  echo "[$(date -u +%H:%M:%SZ)] $MODE done: $(grep DONE /workspace/logs/$CASE-$TAG-$MODE.log | tail -1)" | tee -a $LOG
done
t0=$(date +%s.%N)
$PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/ref1 --ref $R/ref2 --out $R/compare.json > /workspace/logs/$CASE-$TAG-compare.log 2>&1
stamp compare_wall_s "$($PY -c "import time; print(time.time()-$t0)")"
echo "[$(date -u +%H:%M:%SZ)] compare: $(head -1 /workspace/logs/$CASE-$TAG-compare.log)" | tee -a $LOG
# weights -> format v2 (the selective bundles' layout); the v1 bundle is kept as commit-v1
t0=$(date +%s.%N)
rm -rf $R/commit-v2
$PY -m verity_vllm.weights_v2 $R/commit $R/commit-v2 --rows-per-leaf 1 > /workspace/logs/$CASE-$TAG-weights-v2.log 2>&1 || { echo "weights_v2 FAILED" | tee -a $LOG; exit 1; }
stamp weights_v2_wall_s "$($PY -c "import time; print(time.time()-$t0)")"
mv $R/commit $R/commit-v1 && mv $R/commit-v2 $R/commit
echo "[$(date -u +%H:%M:%SZ)] weights v2: $(tail -1 /workspace/logs/$CASE-$TAG-weights-v2.log | cut -c1-200)" | tee -a $LOG
du -sb $R/commit/ru $R/commit/weights $R/commit > $R/du_commit.txt 2>/dev/null
cache_env verify
t0=$(date +%s.%N)
$PY -m verity_vllm_adapter.verify $R/commit --challenge security --q 1 --s 1/4096 --coverage-size 200 --no-negatives --production $R/production --checkpoints $CK --noninterference $R/compare.json --adapter-revision $REV --case $CASE --semantic-profile $SP --out $R/verify-security > /workspace/logs/$CASE-$TAG-verify-security.log 2>&1
stamp verify_wall_s "$($PY -c "import time; print(time.time()-$t0)")"
echo "[$(date -u +%H:%M:%SZ)] verify: $(grep RESULT /workspace/logs/$CASE-$TAG-verify-security.log | tail -1)" | tee -a $LOG
echo "[$(date -u +%H:%M:%SZ)] CONTROL_DONE" | tee -a $LOG
