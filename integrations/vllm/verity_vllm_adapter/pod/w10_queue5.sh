#!/bin/bash
# W10 fifth queue. Why: the first r4 verify (queue 4, 20:51Z) ran the N05 recompute negative that patched
# BOTH copies of the repeated identity (state-identity.md §6) and so accepted its own perturbation -> the
# row is FAIL by the §10 rule on a harness construction error (everything else 178,153/178,153 openings,
# 910/910 + 111/111 relations, 58/58 other negatives). The negative is fixed (unit-scoped patch, two
# variants). Order: r4 re-verify (needs r4's commit values, 33 GB -> before the purge) -> purge r4 ->
# r3 resume at commit with the fresh reference storing values (byte-level cached-vs-fresh over 9,744
# cached positions) -> purge r3 -> r2-512 -> r2-2048 (productions kept). Queue 4 was stopped after its
# r4 record comparison, before it started r3.
# usage: nohup bash w10_queue5.sh <REV> > /workspace/logs/w10_queue5.log 2>&1 &
set -u
REV=$1
S=/workspace/veritor/verity_vllm_adapter/pod/run_state_row.sh
WL=/workspace/results/B1/workload.json
WL3=/workspace/results/B1/workload_r3.json
export PY=/workspace/venv/bin/python
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
purge() {  # values.bin of one finished row (its JSON, indexes and commitments stay)
  D=/workspace/results/B1/$1
  grep -q PIPELINE_DONE $D/pipeline.log 2>/dev/null || { log "not purging $1 (pipeline not done)"; return; }
  find $D/commit/ru $D/ref1/ru $D/ref2/ru $D/fresh/ru -name values.bin -delete 2>/dev/null && echo "purged values of $1" >> $D/pipeline.log
  df -h /workspace | tail -1
}
pgrep -af "w10_queue|run_state_row|verity_vllm_adapter" | grep -v w10_queue5 | cut -c1-120
# R4: re-verify the same commit with the fixed N05 recompute negative (first verify kept aside)
R=/workspace/results/B1/r4; L=/workspace/logs/B1-r4
[ -d $R/verify-n05-both-copies ] || mv $R/verify $R/verify-n05-both-copies
cp $L-verify.log $L-verify-n05-both-copies.log 2>/dev/null
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VERITY_WORKER=W10
ulimit -n 65536
$PY -m verity_vllm_adapter.verify $R/commit --seed 7 --sample-size 1000 --negative-limit 4 --negatives N01,N02,N03,N04,N05,N06,N07,N13,N14 \
  --production $R/production --noninterference $R/compare.json --adapter-revision $REV --case B1 --out $R/verify > $L-verify.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] re-verify (fixed N05 recompute negative; first verify kept as verify-n05-both-copies): $(grep RESULT $L-verify.log | tail -1)" | tee -a $R/pipeline.log
$PY -m verity_vllm_adapter.state_rows recompute-check $R/commit --out $R/recompute_check.json >> $R/pipeline.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] PIPELINE_DONE B1/r4 (re-verify)" | tee -a $R/pipeline.log
log "r4: $(grep -E 're-verify' $R/pipeline.log | tail -1 | cut -c1-200)"
purge r4
# R3: resume at commit (fd limit raised inside run_state_row.sh; fresh reference stores values)
bash $S B1 r3 $REV $WL3 --engine-arg enable_prefix_caching=true --replay-prefix-caching --fresh-ref --replay-engine-arg max_num_batched_tokens=2048 --arena-gb 8 --resume-at-commit
log "r3: $(grep -E 'verify:|FAILED' /workspace/results/B1/r3/pipeline.log | tail -1 | cut -c1-200)"
purge r3
for N in 512 2048; do
  bash $S B1 r2-$N $REV $WL --engine-arg max_num_batched_tokens=$N --engine-arg max_num_seqs=16 --arena-gb 8 --skip-production
  log "r2-$N: $(grep -E 'verify:|FAILED' /workspace/results/B1/r2-$N/pipeline.log | tail -1 | cut -c1-200)"
  purge r2-$N
done
log QUEUE5_DONE
