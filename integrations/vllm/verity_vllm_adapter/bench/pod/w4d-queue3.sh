#!/bin/bash
# W4d queue 3: after queue 2, the Instruct 1.5B checkpoint named in the lane brief (same architecture as B1; cheap).
until grep -q QUEUE2_DONE /workspace/logs/queue.log 2>/dev/null; do sleep 20; done
echo "[$(date -u +%FT%TZ)] queue3 start" >> /workspace/logs/queue.log
cd /workspace/veritor/vllm-poc
bash verity_vllm_adapter/bench/pod/run_matrix.sh /workspace/results/bench \
  Qwen/Qwen2.5-1.5B-Instruct=$(ls -d /workspace/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*) >> /workspace/logs/matrix-1p5i.log 2>&1
echo "[$(date -u +%FT%TZ)] QUEUE3_DONE" >> /workspace/logs/queue.log
