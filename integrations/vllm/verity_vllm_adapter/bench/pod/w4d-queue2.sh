#!/bin/bash
# W4d queue 2: waits for queue 1 (marker in the log, no pgrep), then B7 kernel-profile pass (Task 2 Table 2) and the optional 32B matrix.
until grep -q QUEUE_DONE /workspace/logs/queue.log 2>/dev/null; do sleep 20; done
echo "[$(date -u +%FT%TZ)] queue2 start" >> /workspace/logs/queue.log
cd /workspace/veritor/vllm-poc
# Task 2 Table 2: kernel identity under compiled BI (C1) and eager BI (C2) for Qwen/Qwen2.5-7B (base), warm-up + profiler only
CONFIGS="C1 C2" REPS=0 bash verity_vllm_adapter/bench/pod/run_matrix.sh /workspace/results/bench-b7 \
  Qwen/Qwen2.5-7B=/workspace/hf/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 >> /workspace/logs/matrix-b7.log 2>&1
echo "[$(date -u +%FT%TZ)] b7 profile done" >> /workspace/logs/queue.log
# optional 32B (65 GB bf16 weights; KV budget tight -> achieved concurrency is what it is, recorded)
bash verity_vllm_adapter/bench/pod/run_matrix.sh /workspace/results/bench \
  Qwen/Qwen2.5-32B-Instruct=/workspace/hf/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd:0.93 >> /workspace/logs/matrix-32b.log 2>&1
echo "[$(date -u +%FT%TZ)] QUEUE2_DONE" >> /workspace/logs/queue.log
