#!/bin/bash
# W10 sixth queue. Why: queue 4's purge step (run after its r4 pipeline finished, seconds before it was
# stopped) deleted r4's commit values, so queue 5's r4 re-verify died on a missing values.bin and moved on
# to r3 (running now). This queue waits for r3, then re-runs the r4 commit replay (production's scheduler
# config, 13 GB arena; ref1/ref2 are hash-only and stay) -> compare -> verify with the fixed N05 recompute
# negative -> row checks, then r2-512 and r2-2048 as time allows. The first r4 verify stays under
# r4/verify-n05-both-copies (row B1-1cf9d9d7dc62-20260907T2051Z, collected).
# usage: nohup bash w10_queue6.sh <REV> > /workspace/logs/w10_queue6.log 2>&1 &
set -u
REV=$1
S=/workspace/veritor/verity_vllm_adapter/pod/run_state_row.sh
WL=/workspace/results/B1/workload.json
export PY=/workspace/venv/bin/python
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
purge() {  # values.bin of one finished row (its JSON, indexes and commitments stay)
  D=/workspace/results/B1/$1
  grep -q "PIPELINE_DONE" $D/pipeline.log 2>/dev/null || { log "not purging $1 (pipeline not done)"; return; }
  find $D/commit/ru $D/ref1/ru $D/ref2/ru $D/fresh/ru -name values.bin -delete 2>/dev/null && echo "purged values of $1" >> $D/pipeline.log
  df -h /workspace | tail -1
}
# stop queue 5 (its r3 pipeline keeps running as an orphan); the [.] keeps pkill from matching this script
pkill -f "w10_queue5[.]run.sh"; sleep 1
pgrep -af "w10_queue|run_state_row|verity_vllm_adapter" | grep -v "w10_queue6" | cut -c1-120
log "waiting for the r3 pipeline"
while ! tail -1 /workspace/results/B1/r3/pipeline.log | grep -q "PIPELINE_DONE\|FAILED"; do sleep 30; done
log "r3: $(grep -E 'verify:|FAILED' /workspace/results/B1/r3/pipeline.log | tail -1 | cut -c1-200)"
purge r3
# R4: commit replay again (values were purged), compare, verify (fixed N05), checks
R=/workspace/results/B1/r4
echo "[$(date -u +%H:%M:%SZ)] queue 6: the 20:57Z re-verify above found no values.bin (queue 4 purged r4's values right after its own r4 pipeline finished); commit replay re-run, then verify" >> $R/pipeline.log
bash $S B1 r4 $REV $WL --engine-arg num_gpu_blocks_override=300 --engine-arg max_num_seqs=16 --engine-arg enable_prefix_caching=false --arena-gb 13 --resume-at-commit
log "r4: $(grep -E 'verify:|FAILED' $R/pipeline.log | tail -1 | cut -c1-200)"
purge r4
for N in 512 2048; do
  bash $S B1 r2-$N $REV $WL --engine-arg max_num_batched_tokens=$N --engine-arg max_num_seqs=16 --arena-gb 8 --skip-production
  log "r2-$N: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-$N/pipeline.log | tail -1 | cut -c1-200)"
  purge r2-$N
done
log QUEUE6_DONE
