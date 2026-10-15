#!/usr/bin/env bash
# VLLM-14 scratch: fold rank 0 of circuits' h8 GLM-5.3 TP8 capture alone (~40+ GB), with this tree's patterns.
set -euo pipefail
R=/workspace/fs/glm53-failfast/sweep-auto/20261015T085245Z/glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
export PYTHONPATH="$PWD:$PWD/../..:$PWD/../../tools/research/src"
mkdir -p "$OUT/rank0"
/usr/bin/time -v /workspace/cp/venv312/bin/python -m verity_vllm.pipeline.cli match --worker fold --log "$R/match/capture/log.rank0.jsonl.gz" \
  --profile derived_GLM53_tp8 --out "$OUT/rank0/fold_summary.json" --persist-dir "$OUT/rank0" 2> "$OUT/rank0/time.txt"
