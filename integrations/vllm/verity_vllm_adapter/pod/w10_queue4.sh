#!/bin/bash
# W10 fourth queue. Why: (1) the r3 commit died with EMFILE (965 replay units > the 1024-fd default; the
# commitment builder maps every values.bin) -- resumed at commit with the limit raised; (2) the r4 replay
# under the pod's 2048-token chunk restriction did NOT preempt (production 130 steps / 1 preemption, replay
# 132 steps / 0), so the row is re-replayed under production's own scheduler config (no chunk restriction,
# 13 GB arena: a 4736-token step captures ~12 GB). Then the remaining R2 chunk sizes (productions kept).
# usage: nohup bash w10_queue4.sh <REV> > /workspace/logs/w10_queue4.log 2>&1 &
set -u
REV=$1
S=/workspace/veritor/verity_vllm_adapter/pod/run_state_row.sh
WL=/workspace/results/B1/workload.json
WL3=/workspace/results/B1/workload_r3.json
export PY=/workspace/venv/bin/python
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
purge_prev() {
  for D in /workspace/results/B1/*/; do
    T=$(basename $D); [ "$T" = "$1" ] && continue; [ "$T" = "r3" ] && continue
    grep -q PIPELINE_DONE $D/pipeline.log 2>/dev/null || continue
    [ -d $D/commit/ru ] && find $D/commit/ru $D/ref1/ru $D/ref2/ru $D/fresh/ru -name values.bin -delete 2>/dev/null && echo "purged values of $T" >> $D/pipeline.log
  done
  df -h /workspace | tail -1
}
records() {  # compare-records needs the package on PYTHONPATH (queue 3 ran it from /workspace/logs and it failed silently)
  (cd /workspace/veritor/vllm-poc && PYTHONPATH=. $PY -m verity_vllm_adapter.state_rows compare-records "$@")
}
# stop queue 3 and the r4 pipeline it is running (ref2 of the non-preempting replay)
pkill -f w10_queue3.run.sh; sleep 1
pkill -f "run_state_row.sh B1 r4"; sleep 1
pkill -f "verity_vllm_adapter.replay --case B1"; sleep 8
pgrep -af "verity_vllm_adapter" | cut -c1-120
mv /workspace/results/B1/r4/ref1 /workspace/results/B1/r4/ref1-chunked-no-preemption 2>/dev/null
rm -rf /workspace/results/B1/r4/ref2 /workspace/results/B1/r4/commit
echo "[$(date -u +%H:%M:%SZ)] queue 4: replays under the 2048-token restriction did not preempt (ref1 kept as ref1-chunked-no-preemption); rerun without it" >> /workspace/results/B1/r4/pipeline.log
# R4: production kept (300 blocks, 1 preemption); replay under production's scheduler config
purge_prev r4
bash $S B1 r4 $REV $WL --engine-arg num_gpu_blocks_override=300 --engine-arg max_num_seqs=16 --engine-arg enable_prefix_caching=false --arena-gb 13 --skip-production
log "r4: $(grep -E 'verify:|FAILED' /workspace/results/B1/r4/pipeline.log | tail -1)"
records base=/workspace/results/B1/base/production/production.jsonl r4=/workspace/results/B1/r4/production/production.jsonl --out /workspace/results/B1/r4_compare_records.json
log "r4 record comparison: $(tail -c 200 /workspace/results/B1/r4_compare_records.json | tr -d '\n')"
# R3: resume at commit (fd limit raised inside run_state_row.sh)
purge_prev r3
bash $S B1 r3 $REV $WL3 --engine-arg enable_prefix_caching=true --replay-prefix-caching --fresh-ref --replay-engine-arg max_num_batched_tokens=2048 --arena-gb 8 --resume-at-commit
log "r3: $(grep -E 'verify:|FAILED' /workspace/results/B1/r3/pipeline.log | tail -1)"
for N in 512 2048; do
  purge_prev r2-$N
  bash $S B1 r2-$N $REV $WL --engine-arg max_num_batched_tokens=$N --engine-arg max_num_seqs=16 --arena-gb 8 --skip-production
  log "r2-$N: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-$N/pipeline.log | tail -1)"
done
log QUEUE4_DONE
