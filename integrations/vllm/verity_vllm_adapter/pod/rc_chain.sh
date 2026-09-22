#!/bin/bash
# RC: experiment 1 on the pod end to end, sequentially on the one GPU: full-preparation control, then the
# selective runs K in {2,4,8,16} (K=1 ran first as the smoke test, revision 249c645, and is re-run last at
# this revision), each followed by the batch-invariance comparison against the control.
# usage: rc_chain.sh <REV>
set -u
REV=$1
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
PY=${PY:-/workspace/venv-cu129/bin/python}
R=/workspace/rc/results/B1
POD=verity_vllm_adapter/pod
echo "[$(date -u +%H:%M:%SZ)] CHAIN start rev=$REV" | tee -a $R/chain.log
if [ ! -f $R/control/verify-security/verify.json ]; then
  bash $POD/run_control.sh $REV >> $R/chain.log 2>&1
fi
for K in 2 4 8 16 1; do
  TAG=k$K; [ $K = 1 ] && TAG=k1-rerun
  bash $POD/run_selective.sh $K $REV --tag $TAG >> $R/chain.log 2>&1
  $PY -m verity_vllm_adapter.compare_selected --selective $R/selective-$TAG --control $R/control/commit --out $R/selective-$TAG.verify/compare_selected.json \
      > /workspace/logs/rc-compare-$TAG.log 2>&1
  echo "[$(date -u +%H:%M:%SZ)] compare_selected $TAG: $(tail -1 /workspace/logs/rc-compare-$TAG.log | cut -c1-300)" | tee -a $R/chain.log
  du -sb $R/selective-$TAG/ru $R/selective-$TAG/weights $R/selective-$TAG > $R/selective-$TAG.verify/du.txt 2>/dev/null
done
$PY -m verity_vllm_adapter.compare_selected --selective $R/selective-k1 --control $R/control/commit --out $R/selective-k1.verify/compare_selected.json > /workspace/logs/rc-compare-k1.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] compare_selected k1: $(tail -1 /workspace/logs/rc-compare-k1.log | cut -c1-300)" | tee -a $R/chain.log
echo "[$(date -u +%H:%M:%SZ)] CHAIN_DONE" | tee -a $R/chain.log
