#!/bin/bash
# W4d sequential GPU queue: 7B/14B matrix -> Task 2 A/B recordings. No waiting on other processes.
cd /workspace/veritor/vllm-poc
echo "[$(date -u +%FT%TZ)] queue start" >> /workspace/logs/queue.log
bash verity_vllm_adapter/bench/pod/run_matrix.sh /workspace/results/bench \
  Qwen/Qwen2.5-7B-Instruct=/workspace/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  Qwen/Qwen2.5-14B-Instruct=/workspace/hf/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 >> /workspace/logs/matrix-2.log 2>&1
echo "[$(date -u +%FT%TZ)] matrix-2 done" >> /workspace/logs/queue.log
REV=220f2bb bash /workspace/w4d-task2.sh >> /workspace/logs/task2-driver.log 2>&1
echo "[$(date -u +%FT%TZ)] QUEUE_DONE" >> /workspace/logs/queue.log
