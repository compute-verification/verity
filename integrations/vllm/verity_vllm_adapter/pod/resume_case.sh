#!/bin/bash
# Resume a run_case.sh pipeline after a failed commit-mode replay: commit -> compare -> verify (refs kept).
# usage: resume_case.sh <CASE> <TAG> <REV> [--arena-gb N] [--seed N] [--replay-engine-arg K=V]
set -u
CASE=$1; TAG=$2; REV=$3; shift 3
ARENA=8; SEED=7; REA=()
while [ $# -gt 0 ]; do case "$1" in --arena-gb) ARENA=$2; shift;; --seed) SEED=$2; shift;; --replay-engine-arg) REA+=(--engine-arg "$2"); shift;; esac; shift; done
cd /workspace/veritor/vllm-poc
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
PY=${PY:-/workspace/venv/bin/python}
ulimit -n 65536
R=/workspace/results/$CASE/$TAG; L=/workspace/logs/$CASE-$TAG
CK=manifests/checkpoints.json
echo "[$(date -u +%H:%M:%SZ)] resume $CASE/$TAG at commit, rev=$REV arena=${ARENA}G" | tee -a $R/pipeline.log
rm -rf $R/commit
$PY -m verity_vllm_adapter.replay --case $CASE --checkpoints $CK --production $R/production --out $R/commit --mode commit --weights-values store "${REA[@]}" --adapter-revision $REV --arena-gb $ARENA > $L-commit.log 2>&1 || { echo "commit FAILED (resume)" | tee -a $R/pipeline.log; exit 1; }
echo "[$(date -u +%H:%M:%SZ)] commit done: $(grep DONE $L-commit.log | tail -1)" | tee -a $R/pipeline.log
$PY -m verity_vllm_adapter.compare --commit $R/commit --ref $R/ref1 --ref $R/ref2 --out $R/compare.json > $L-compare.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] compare: $(head -1 $L-compare.log)" | tee -a $R/pipeline.log
$PY -m verity_vllm_adapter.verify $R/commit --seed $SEED --sample-size 1000 --negative-limit 4 --production $R/production --noninterference $R/compare.json --adapter-revision $REV --case $CASE --out $R/verify > $L-verify.log 2>&1
echo "[$(date -u +%H:%M:%SZ)] verify: $(grep RESULT $L-verify.log | tail -1)" | tee -a $R/pipeline.log
echo "[$(date -u +%H:%M:%SZ)] PIPELINE_DONE $CASE/$TAG" | tee -a $R/pipeline.log
