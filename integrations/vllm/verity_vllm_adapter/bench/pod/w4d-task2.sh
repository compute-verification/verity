#!/bin/bash
# W4d Task 2: compiled-vs-eager production A/B on the H100 (adapter recorder, production-only, fresh engine per run).
REV=${REV:-220f2bb}
cd /workspace/veritor/vllm-poc
echo "[$(date -u +%FT%TZ)] task2 start rev=$REV" >> /workspace/logs/task2.log
for CASE in B1 B7; do
  bash verity_vllm_adapter/pod/run_case.sh $CASE h100-compiled $REV --production-only >> /workspace/logs/task2.log 2>&1
  bash verity_vllm_adapter/pod/run_case.sh $CASE h100-eager $REV --production-only --enforce-eager \
      --compare-tokens /workspace/results/$CASE/h100-compiled/production/production.jsonl >> /workspace/logs/task2.log 2>&1
  # second fresh-process repetition of each profile (same-device bit-reproducibility, §11 "three fresh-process repetitions" budgeted to two here)
  bash verity_vllm_adapter/pod/run_case.sh $CASE h100-compiled-r2 $REV --production-only \
      --compare-tokens /workspace/results/$CASE/h100-compiled/production/production.jsonl >> /workspace/logs/task2.log 2>&1
  bash verity_vllm_adapter/pod/run_case.sh $CASE h100-eager-r2 $REV --production-only --enforce-eager \
      --compare-tokens /workspace/results/$CASE/h100-eager/production/production.jsonl >> /workspace/logs/task2.log 2>&1
done
echo "[$(date -u +%FT%TZ)] TASK2_DONE" >> /workspace/logs/task2.log
