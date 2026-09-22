#!/bin/bash
# W10 third queue (queue 2's watcher fired on the "r4 probe" line of queue 1 and stopped it during the
# 272-block probe; queue 2 is now producing the quick r2-512 / r2-2048 rows). Takes over after the
# r2-2048 quick row and runs the remaining rows in priority order, one GPU, strictly sequential:
#   r4 probe 272 (production only) -> r2-256 (full) -> R2 record comparison -> r3 (full) -> r4 (full) ->
#   r2-512 (full, production kept) -> r2-2048 (full, production kept)
# usage: nohup bash w10_queue3.sh <REV> > /workspace/logs/w10_queue3.log 2>&1 &
set -u
REV=$1
S=/workspace/veritor/verity_vllm_adapter/pod/run_state_row.sh
WL=/workspace/results/B1/workload.json
WL3=/workspace/results/B1/workload_r3.json

export PY=/workspace/venv/bin/python
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
# 150 GB disk, ~31 GB of committed values per full row: before a full row starts, drop values.bin of every
# finished row except r3 (its bundle is the one kept for §16.8); index/boundary/commitments/verify stay.
purge_prev() {
  for D in /workspace/results/B1/*/; do
    T=$(basename $D); [ "$T" = "$1" ] && continue; [ "$T" = "r3" ] && continue
    grep -q PIPELINE_DONE $D/pipeline.log 2>/dev/null || continue
    [ -d $D/commit/ru ] && find $D/commit/ru $D/ref1/ru $D/ref2/ru $D/fresh/ru -name values.bin -delete 2>/dev/null && echo "purged values of $T" >> $D/pipeline.log
  done
  df -h /workspace | tail -1
}
# queue 2's bash was stopped by hand; its r2-512 --quick child (production done, ref1 running) finishes on its own
log "waiting for the r2-512 quick row"
while ! grep -qE 'QUICK_DONE|FAILED' /workspace/results/B1/r2-512/pipeline.log 2>/dev/null; do sleep 15; done
log "r2-512 quick: $(grep -E 'ref1 done|FAILED' /workspace/results/B1/r2-512/pipeline.log | tail -1)"
while pgrep -f "verity_vllm_adapter.replay --case B1" > /dev/null; do sleep 5; done
bash $S B1 r2-2048 $REV $WL --engine-arg max_num_batched_tokens=2048 --engine-arg max_num_seqs=16 --arena-gb 8 --quick
log "r2-2048 quick: $(grep -E 'ref1 done|FAILED' /workspace/results/B1/r2-2048/pipeline.log | tail -1)"
# R4 witness probe 2 (the 300-block probe preempted once)
bash $S B1 r4-probe-272 $REV $WL --engine-arg num_gpu_blocks_override=272 --engine-arg max_num_seqs=16 --engine-arg enable_prefix_caching=false --production-only --arena-gb 8
log "r4 probe 272: $(grep -E 'production done|FAILED' /workspace/results/B1/r4-probe-272/pipeline.log | tail -1)"
# R2 chunked prefill at 256 (full row)
purge_prev r2-256
bash $S B1 r2-256 $REV $WL --engine-arg max_num_batched_tokens=256 --engine-arg max_num_seqs=16 --arena-gb 8
log "r2-256: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-256/pipeline.log | tail -1)"
$PY -m verity_vllm_adapter.state_rows compare-records base=/workspace/results/B1/base/production/production.jsonl \
  r2-256=/workspace/results/B1/r2-256/production/production.jsonl r2-512=/workspace/results/B1/r2-512/production/production.jsonl \
  r2-2048=/workspace/results/B1/r2-2048/production/production.jsonl --out /workspace/results/B1/r2_compare_records.json
log "r2 record comparison: $(tail -c 300 /workspace/results/B1/r2_compare_records.json | tr -d '\n' | cut -c1-200)"
# R3 prefix-cache hits: cached replay under the cross-request rule + one fresh reference
purge_prev r3
bash $S B1 r3 $REV $WL3 --engine-arg enable_prefix_caching=true --replay-prefix-caching --fresh-ref --replay-engine-arg max_num_batched_tokens=2048 --arena-gb 8
log "r3: $(grep -E 'verify:|FAILED' /workspace/results/B1/r3/pipeline.log | tail -1)"
# R4: the probe budget with the most preemptions (both > 0 expected; 300 gave 1)
B4=""; PMAX=0
for B in 272 300; do
  P=$(grep -E 'production done' /workspace/results/B1/r4-probe-$B/pipeline.log 2>/dev/null | grep -oE 'preemptions=[0-9]+' | tail -1 | cut -d= -f2)
  if [ -n "$P" ] && [ "$P" -gt "$PMAX" ]; then B4=$B; PMAX=$P; fi
done
if [ -n "$B4" ]; then
  purge_prev r4
  bash $S B1 r4 $REV $WL --engine-arg num_gpu_blocks_override=$B4 --engine-arg max_num_seqs=16 --engine-arg enable_prefix_caching=false --replay-engine-arg max_num_batched_tokens=2048 --arena-gb 8
  log "r4 (blocks=$B4, probe preemptions=$PMAX): $(grep -E 'verify:|FAILED' /workspace/results/B1/r4/pipeline.log | tail -1)"
  $PY -m verity_vllm_adapter.state_rows compare-records base=/workspace/results/B1/base/production/production.jsonl \
    r4=/workspace/results/B1/r4/production/production.jsonl --out /workspace/results/B1/r4_compare_records.json
  log "r4 record comparison: $(tail -c 300 /workspace/results/B1/r4_compare_records.json | tr -d '\n' | cut -c1-200)"
else
  log "r4: no probe preempted; row NOT RUN"
fi
for N in 512 2048; do
  purge_prev r2-$N
  bash $S B1 r2-$N $REV $WL --engine-arg max_num_batched_tokens=$N --engine-arg max_num_seqs=16 --arena-gb 8 --skip-production
  log "r2-$N: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-$N/pipeline.log | tail -1)"
done
log QUEUE3_DONE
