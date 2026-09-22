#!/bin/bash
# W10 run queue (one GPU, strictly sequential): waits for the B1 eager baseline, then the state rows.
# usage: nohup bash w10_queue.sh <REV> > /workspace/logs/w10_queue.log 2>&1 &
set -u
REV=$1
S=/workspace/veritor/verity_vllm_adapter/pod/run_state_row.sh
WL=/workspace/results/B1/workload.json
WL3=/workspace/results/B1/workload_r3.json
export PY=/workspace/venv/bin/python
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
# 150 GB disk, ~30 GB of committed values per full row: before a new full row starts, drop the raw
# values (values.bin) of every finished row except the one named in KEEP; index/boundary/commitments/
# verify outputs stay (bundle.json records sha256 + pod path of what was there).
KEEP=""
purge_prev() {
  for D in /workspace/results/B1/*/; do
    T=$(basename $D); [ "$T" = "$1" ] && continue; [ "$T" = "$KEEP" ] && continue; [ "$T" = "base" ] && continue
    grep -q PIPELINE_DONE $D/pipeline.log 2>/dev/null || continue
    [ -d $D/commit/ru ] && find $D/commit/ru $D/ref1/ru $D/ref2/ru $D/fresh/ru -name values.bin -delete 2>/dev/null && echo "purged values of $T" >> $D/pipeline.log
  done
  df -h /workspace | tail -1
}
# the baseline's production (old tree, 0e1c233) is kept; its replays re-run on the W10 tree with the
# replay-side chunk restriction (commit mode at 8192 tokens/step was OOM-killed by the 31 GB container limit)
rsync -a --delete --exclude .git /workspace/veritor-dev/ /workspace/veritor/ && log "code synced"
rm -rf /workspace/results/B1/base/ref1 /workspace/results/B1/base/ref2 /workspace/results/B1/base/commit /workspace/results/B1/base/verify
bash $S B1 base $REV $WL --skip-production --replay-engine-arg max_num_batched_tokens=2048 --arena-gb 8
log "baseline: $(grep -E 'verify:|FAILED' /workspace/results/B1/base/pipeline.log | tail -1)"
# R4 witness probes (production only): find a KV budget that preempts
for B in 300 272; do
  bash $S B1 r4-probe-$B $REV $WL --engine-arg num_gpu_blocks_override=$B --engine-arg max_num_seqs=16 --engine-arg enable_prefix_caching=false --production-only --arena-gb 8
  log "r4 probe $B: $(grep -E 'production done|FAILED' /workspace/results/B1/r4-probe-$B/pipeline.log | tail -1)"
done
# R2 chunked prefill, one chunk size first (validates the override plumbing end to end)
purge_prev r2-256
bash $S B1 r2-256 $REV $WL --engine-arg max_num_batched_tokens=256 --engine-arg max_num_seqs=16 --arena-gb 8
log "r2-256: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-256/pipeline.log | tail -1)"
# R3 prefix-cache hits: cached replay under the cross-request rule + one fresh reference
purge_prev r3
bash $S B1 r3 $REV $WL3 --engine-arg enable_prefix_caching=true --replay-prefix-caching --fresh-ref --replay-engine-arg max_num_batched_tokens=2048 --arena-gb 8
KEEP=r3
log "r3: $(grep -E 'verify:|FAILED' /workspace/results/B1/r3/pipeline.log | tail -1)"
# R4: the smallest budget that preempted in the probes
B4=""
for B in 272 300; do
  P=$(grep -E 'production done' /workspace/results/B1/r4-probe-$B/pipeline.log 2>/dev/null | grep -oE 'preemptions=[0-9]+' | cut -d= -f2)
  if [ -n "$P" ] && [ "$P" -gt 0 ]; then B4=$B; break; fi
done
if [ -n "$B4" ]; then
  purge_prev r4
  bash $S B1 r4 $REV $WL --engine-arg num_gpu_blocks_override=$B4 --engine-arg max_num_seqs=16 --engine-arg enable_prefix_caching=false --replay-engine-arg max_num_batched_tokens=2048 --arena-gb 8
  log "r4 (blocks=$B4): $(grep -E 'verify:|FAILED' /workspace/results/B1/r4/pipeline.log | tail -1)"
else
  log "r4: no probe preempted; row NOT RUN"
fi
for N in 512 2048; do
  purge_prev r2-$N
  bash $S B1 r2-$N $REV $WL --engine-arg max_num_batched_tokens=$N --engine-arg max_num_seqs=16 --arena-gb 8
  log "r2-$N: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-$N/pipeline.log | tail -1)"
done
log QUEUE_DONE
