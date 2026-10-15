#!/usr/bin/env bash
# VLLM-14 scratch: re-fold circuits' h8 GLM-5.3 TP8 capture (sweep-auto/20261015T085245Z) with this tree's fold patterns.
set -euo pipefail
R=/workspace/fs/glm53-failfast/sweep-auto/20261015T085245Z/glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
PY=/workspace/cp/venv312/bin/python
args=(--capture "$R/match/capture" --profile derived_GLM53_tp8)
for r in 0 1 2 3 4 5 6 7; do
  args+=(--build-rank "$R/build/rank$r/build_request_LP256_T31" --gprog-rank "$r=$R/build/rank$r/build_workload/workload_program.json")
done
args+=(--workload workloads/glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager.json --tp2-match "$R/match/tp2_match.json"
       --within-step dag --rank-workers 0 --python "$PY" --timeout 7200 --out "$OUT/fold_match")
export PYTHONPATH="$PWD:$PWD/../..:$PWD/../../tools/research/src"
"$PY" -m verity_vllm.pipeline.cli tp-fold-match "${args[@]}"
