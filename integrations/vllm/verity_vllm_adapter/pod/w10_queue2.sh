#!/bin/bash
# W10 second queue: takes over from w10_queue.sh once its r4 row is logged (pod terminates 01:35Z), so the
# remaining R2 chunk sizes get their production record + reference replay (witness, token/logprob bit
# comparison, kv_unresolved) BEFORE the 60-minute commit/verify tails, which then run while time remains.
# usage: nohup bash w10_queue2.sh <REV> > /workspace/logs/w10_queue2.log 2>&1 &
set -u
REV=$1
S=/workspace/veritor/verity_vllm_adapter/pod/run_state_row.sh
WL=/workspace/results/B1/workload.json
Q1=/workspace/logs/w10_queue.log
export PY=/workspace/venv/bin/python
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
purge_prev() {
  for D in /workspace/results/B1/*/; do
    T=$(basename $D); [ "$T" = "$1" ] && continue; [ "$T" = "r3" ] && continue; [ "$T" = "base" ] && continue
    grep -q PIPELINE_DONE $D/pipeline.log 2>/dev/null || continue
    [ -d $D/commit/ru ] && find $D/commit/ru $D/ref1/ru $D/ref2/ru $D/fresh/ru -name values.bin -delete 2>/dev/null && echo "purged values of $T" >> $D/pipeline.log
  done
  df -h /workspace | tail -1
}
log "waiting for queue 1 to log its r4 row"
while ! grep -qE '^\[[0-9:Z]+\] r4' $Q1 2>/dev/null; do sleep 30; done
log "queue 1 r4 row logged: $(grep -E '^\[[0-9:Z]+\] r4' $Q1 | tail -1)"
# stop queue 1 (it would start the full r2-512 pipeline next) and whatever it just launched
pkill -f w10_queue.run.sh; sleep 1
pkill -f "run_state_row.sh B1 r2-512" ; pkill -f "run_state_row.sh B1 r2-2048"; sleep 1
pkill -f "verity_vllm_adapter.production --case B1" ; pkill -f "verity_vllm_adapter.replay --case B1"; sleep 5
pgrep -af "verity_vllm_adapter" | cut -c1-120
rm -rf /workspace/results/B1/r2-512 /workspace/results/B1/r2-2048
for N in 512 2048; do
  bash $S B1 r2-$N $REV $WL --engine-arg max_num_batched_tokens=$N --engine-arg max_num_seqs=16 --arena-gb 8 --quick
  log "r2-$N quick: $(grep -E 'ref1 done|FAILED' /workspace/results/B1/r2-$N/pipeline.log | tail -1)"
done
$PY -m verity_vllm_adapter.state_rows compare-records base=/workspace/results/B1/base/production/production.jsonl \
  r2-256=/workspace/results/B1/r2-256/production/production.jsonl r2-512=/workspace/results/B1/r2-512/production/production.jsonl \
  r2-2048=/workspace/results/B1/r2-2048/production/production.jsonl --out /workspace/results/B1/r2_compare_records.json
log "r2 record comparison written"
for N in 512 2048; do
  purge_prev r2-$N
  bash $S B1 r2-$N $REV $WL --engine-arg max_num_batched_tokens=$N --engine-arg max_num_seqs=16 --arena-gb 8 --skip-production
  log "r2-$N: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-$N/pipeline.log | tail -1)"
done
log QUEUE2_DONE
