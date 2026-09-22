#!/bin/bash
# Re-run sampler.select over the existing replay dumps with an explicit top-k tie rule, then N09/N08.
#   nightly (Triton top-k at every batch size):  W3_VENV=/workspace/venv-nightly TIES=exact_k     bash .../run_reanalyze.sh
#   0.28.0  (Triton at >= 8 rows, torch below):  W3_VENV=/workspace/venv         TIES=from_batch:8 bash .../run_reanalyze.sh
# Writes select_<name>_<impl>.json (primary rule; overwrites) and select_<name>_<impl>_unpinned.json
# (abstaining checker: certified only where exact_k and keep_all agree).
set -uo pipefail
source "$(dirname "$0")/env.sh"
TIES=${TIES:-exact_k}
export VLLM_ENABLE_V1_MULTIPROCESSING=0

analyze() {  # name impl ties suffix
  local name=$1 impl=$2 ties=$3 suffix=$4
  local out=$W3_RESULTS/analysis/select_${name}_${impl}${suffix}.json
  local log=$W3_RESULTS/logs/analyze_${name}_${impl}${suffix}.log
  local prod=$W3_RESULTS/prod/${name%_at_c*}.json   # cross-batch dumps (<name>_at_c<cap>) replay the record <name>
  [ -s "$W3_RESULTS/replay/logits_$name/index.json" ] || { echo "[skip] no dump for $name"; return; }
  echo "[$(date -u +%H:%M:%SZ)] analyze $name $impl ties=$ties"
  python -m verity_vllm_sampler.analyze --production "$prod" --logits-dir "$W3_RESULTS/replay/logits_$name" \
      --out "$out" --same-runtime --topk-topp-impl "$impl" --topk-ties "$ties" > "$log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] done   analyze $name $impl$suffix rc=$? $(grep ANALYZE_DONE "$log" | cut -c1-320)"
}

for name in topk_topp_s1_c64_r0 temp_s1_c64_r0 greedy_c64_r0 topk_topp_s2_c64_r0 temp_s2_c64_r0 topk_topp_s3_c64_r0 temp_s3_c64_r0 \
            topk_topp_s1_c8_r0 topk_topp_s1_c64_r0_at_c8 greedy_c64_r0_at_c8 temp_s1_c64_r0_at_c8 topk_topp_s1_c64_r0_at_c1; do
  analyze $name auto "$TIES" ""
done
for name in topk_topp_s1_c64_r0 topk_topp_s2_c64_r0 topk_topp_s3_c64_r0; do
  analyze $name triton "$TIES" ""
  analyze $name auto unpinned _unpinned
done
for name in topk_topp_s1_c64_r0 temp_s1_c64_r0; do
  python -m verity_vllm_sampler.negative n09 --production "$W3_RESULTS/prod/$name.json" --logits-dir "$W3_RESULTS/replay/logits_$name" \
      --out "$W3_RESULTS/negative/n09_$name.json" > "$W3_RESULTS/logs/n09_$name.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] n09 $name rc=$? $(grep N09_DONE "$W3_RESULTS/logs/n09_$name.log" | cut -c1-300)"
done
rm -rf "$W3_RESULTS/negative/n08_topk_topp_s1_i6_k5" "$W3_RESULTS/negative/n08_greedy_i2_k3"
python -m verity_vllm_sampler.negative n08 --production "$W3_RESULTS/prod/topk_topp_s1_c64_r0.json" --honest-logits "$W3_RESULTS/replay/logits_topk_topp_s1_c64_r0" \
    --index 6 --k 5 --out-dir "$W3_RESULTS/negative/n08_topk_topp_s1_i6_k5" > "$W3_RESULTS/logs/n08_topk_topp.log" 2>&1
echo "[$(date -u +%H:%M:%SZ)] n08 topk_topp rc=$? $(grep N08_DONE "$W3_RESULTS/logs/n08_topk_topp.log" | cut -c1-400)"
python -m verity_vllm_sampler.negative n08 --production "$W3_RESULTS/prod/greedy_c64_r0.json" --honest-logits "$W3_RESULTS/replay/logits_greedy_c64_r0" \
    --index 2 --k 3 --out-dir "$W3_RESULTS/negative/n08_greedy_i2_k3" > "$W3_RESULTS/logs/n08_greedy.log" 2>&1
echo "[$(date -u +%H:%M:%SZ)] n08 greedy rc=$? $(grep N08_DONE "$W3_RESULTS/logs/n08_greedy.log" | cut -c1-400)"
echo "REANALYZE_DONE $(date -u +%H:%M:%SZ)"
