#!/bin/bash
# Replay-reference logits dumps + sampler.select analysis + N08/N09 for every recorded variant/seed.
#   setsid nohup bash verity_vllm_sampler/pod/run_replay_batch.sh > $W3_RESULTS/logs/replay_batch.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$W3_RESULTS/replay" "$W3_RESULTS/analysis" "$W3_RESULTS/negative"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

replay() {  # name
  local name=$1
  local dir=$W3_RESULTS/replay/logits_$name
  if [ -s "$dir/index.json" ]; then echo "[skip] replay $name"; return; fi
  echo "[$(date -u +%H:%M:%SZ)] replay $name"
  python -m verity_vllm_sampler.replay_logits --production "$W3_RESULTS/prod/$name.json" --out-dir "$dir" \
      > "$W3_RESULTS/logs/replay_$name.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] done   replay $name rc=$? $(grep REPLAY_DONE "$W3_RESULTS/logs/replay_$name.log" | cut -c1-200)"
}
analyze() {  # name impl
  local name=$1 impl=$2
  local out=$W3_RESULTS/analysis/select_${name}_$impl.json
  if [ -s "$out" ]; then echo "[skip] analyze $name $impl"; return; fi
  echo "[$(date -u +%H:%M:%SZ)] analyze $name $impl"
  python -m verity_vllm_sampler.analyze --production "$W3_RESULTS/prod/$name.json" --logits-dir "$W3_RESULTS/replay/logits_$name" \
      --out "$out" --same-runtime --topk-topp-impl "$impl" > "$W3_RESULTS/logs/analyze_${name}_$impl.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] done   analyze $name $impl rc=$? $(grep ANALYZE_DONE "$W3_RESULTS/logs/analyze_${name}_$impl.log" | cut -c1-260)"
}

for name in topk_topp_s1_c64_r0 temp_s1_c64_r0 greedy_c64_r0 topk_topp_s2_c64_r0 temp_s2_c64_r0 topk_topp_s3_c64_r0 temp_s3_c64_r0; do
  replay $name
  analyze $name auto
done
# the Triton top-k/top-p path (what production used at batch >= 8) as the same_runtime reference
for name in topk_topp_s1_c64_r0 topk_topp_s2_c64_r0 topk_topp_s3_c64_r0; do
  analyze $name triton
done
# N09 over the honest dumps (CPU)
for name in topk_topp_s1_c64_r0 temp_s1_c64_r0; do
  python -m verity_vllm_sampler.negative n09 --production "$W3_RESULTS/prod/$name.json" --logits-dir "$W3_RESULTS/replay/logits_$name" \
      --out "$W3_RESULTS/negative/n09_$name.json" > "$W3_RESULTS/logs/n09_$name.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] n09 $name rc=$? $(grep N09_DONE "$W3_RESULTS/logs/n09_$name.log" | cut -c1-300)"
done
# N08: forge one token, replay continues, select must reject at k (stochastic and greedy)
python -m verity_vllm_sampler.negative n08 --production "$W3_RESULTS/prod/topk_topp_s1_c64_r0.json" --honest-logits "$W3_RESULTS/replay/logits_topk_topp_s1_c64_r0" \
    --index 6 --k 5 --out-dir "$W3_RESULTS/negative/n08_topk_topp_s1_i6_k5" > "$W3_RESULTS/logs/n08_topk_topp.log" 2>&1
echo "[$(date -u +%H:%M:%SZ)] n08 topk_topp rc=$? $(grep N08_DONE "$W3_RESULTS/logs/n08_topk_topp.log" | cut -c1-400)"
python -m verity_vllm_sampler.negative n08 --production "$W3_RESULTS/prod/greedy_c64_r0.json" --honest-logits "$W3_RESULTS/replay/logits_greedy_c64_r0" \
    --index 2 --k 3 --out-dir "$W3_RESULTS/negative/n08_greedy_i2_k3" > "$W3_RESULTS/logs/n08_greedy.log" 2>&1
echo "[$(date -u +%H:%M:%SZ)] n08 greedy rc=$? $(grep N08_DONE "$W3_RESULTS/logs/n08_greedy.log" | cut -c1-400)"
# replay under a different batch composition (cap 8) of the same recorded trajectory: are the logits rows identical?
replay topk_topp_s1_c8_r0
analyze topk_topp_s1_c8_r0 auto
echo "REPLAY_BATCH_DONE $(date -u +%H:%M:%SZ)"
