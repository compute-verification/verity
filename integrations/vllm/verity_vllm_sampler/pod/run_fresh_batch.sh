#!/bin/bash
# Fresh-prefill production records (contract §15.4; lead finding 10:45Z) + their trace replays + sampler.select
# + negatives, on the POC pin (nightly).  Requests 11 and 13 of the workload share one leading 16-token block,
# so the pass is split over two fresh engines (main: all but 13; solo: 13); both warm up on throwaway prompts
# first and assert num_cached_tokens == 0 for every recorded request.
#   W3_VENV=/workspace/venv-nightly setsid nohup bash verity_vllm_sampler/pod/run_fresh_batch.sh > $W3_RESULTS/logs/fresh_batch.log 2>&1 < /dev/null &
set -uo pipefail
source "$(dirname "$0")/env.sh"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
SOLO=${SOLO:-13}
WARM=${WARM:-2}
mkdir -p "$W3_RESULTS/prod_parts" "$W3_RESULTS/replay" "$W3_RESULTS/analysis" "$W3_RESULTS/negative" "$W3_RESULTS/logs"
ts() { date -u +%H:%M:%SZ; }

prod_part() {  # name suffix extra-args...
  local name=$1 suffix=$2; shift 2
  local out=$W3_RESULTS/prod_parts/${name}_$suffix.json
  if [ -s "$out" ]; then echo "[skip] prod $name $suffix"; return; fi
  echo "[$(ts)] prod $name $suffix"
  VLLM_ENABLE_V1_MULTIPROCESSING=1 python -m verity_vllm_sampler.production --workload "$W3_WORKLOAD" --out "$out" \
      --warmup-throwaway "$WARM" --assert-no-cached "$@" > "$W3_RESULTS/logs/prod_${name}_$suffix.log" 2>&1
  echo "[$(ts)] done  rc=$? $(grep -E 'PRODUCTION_DONE|ASSERT_NO_CACHED_FAILED' "$W3_RESULTS/logs/prod_${name}_$suffix.log" | cut -c1-260)"
}
fresh_prod() {  # name variant-args...
  local name=$1; shift
  prod_part "$name" main --exclude "$SOLO" "$@"
  prod_part "$name" solo$SOLO --only "$SOLO" "$@"
  local earlier=$W3_RESULTS/prod/${name%_fresh}_r0.json
  python -m verity_vllm_sampler.fresh merge-prod --main "$W3_RESULTS/prod_parts/${name}_main.json" \
      --solo "$W3_RESULTS/prod_parts/${name}_solo$SOLO.json" --out "$W3_RESULTS/prod/$name.json" \
      $( [ -s "$earlier" ] && echo --compare-to "$earlier" ) 2>&1 | tail -1 | cut -c1-400
}
replay_part() {  # name suffix extra-args...
  local name=$1 suffix=$2; shift 2
  local dir=$W3_RESULTS/replay/logits_${name}_$suffix
  if [ -s "$dir/index.json" ]; then echo "[skip] replay $name $suffix"; return; fi
  echo "[$(ts)] replay $name $suffix"
  python -m verity_vllm_sampler.replay_logits --production "$W3_RESULTS/prod/$name.json" --out-dir "$dir" \
      --warmup-throwaway "$WARM" "$@" > "$W3_RESULTS/logs/replay_${name}_$suffix.log" 2>&1
  echo "[$(ts)] done  rc=$? $(grep REPLAY_DONE "$W3_RESULTS/logs/replay_${name}_$suffix.log" | cut -c1-300)"
}
fresh_replay() {  # name
  local name=$1
  replay_part "$name" main --exclude "$SOLO"
  replay_part "$name" solo$SOLO --only "$SOLO"
  python -m verity_vllm_sampler.fresh merge-dumps --main "$W3_RESULTS/replay/logits_${name}_main" \
      --solo "$W3_RESULTS/replay/logits_${name}_solo$SOLO" --out "$W3_RESULTS/replay/logits_$name" 2>&1 | tail -1 | cut -c1-300
}
analyze() {  # name impl ties suffix
  local name=$1 impl=$2 ties=$3 suffix=$4
  local out=$W3_RESULTS/analysis/select_${name}_${impl}${suffix}.json
  if [ -s "$out" ]; then echo "[skip] analyze $name $impl$suffix"; return; fi
  echo "[$(ts)] analyze $name $impl ties=$ties"
  python -m verity_vllm_sampler.analyze --production "$W3_RESULTS/prod/$name.json" --logits-dir "$W3_RESULTS/replay/logits_$name" \
      --out "$out" --same-runtime --topk-topp-impl "$impl" --topk-ties "$ties" > "$W3_RESULTS/logs/analyze_${name}_${impl}${suffix}.log" 2>&1
  echo "[$(ts)] done  rc=$? $(grep ANALYZE_DONE "$W3_RESULTS/logs/analyze_${name}_${impl}${suffix}.log" | cut -c1-360)"
}

NAMES="greedy_c64_fresh temp_s1_c64_fresh topk_topp_s1_c64_fresh temp_s2_c64_fresh topk_topp_s2_c64_fresh temp_s3_c64_fresh topk_topp_s3_c64_fresh"
# ---- production (fresh engines, one pass, throwaway warm-up, no cached prefix)
fresh_prod greedy_c64_fresh       --variant greedy             --max-num-seqs 64 --rep 0
fresh_prod temp_s1_c64_fresh      --variant temp      --seed 1 --max-num-seqs 64 --rep 0
fresh_prod topk_topp_s1_c64_fresh --variant topk_topp --seed 1 --max-num-seqs 64 --rep 0
fresh_prod temp_s2_c64_fresh      --variant temp      --seed 2 --max-num-seqs 64 --rep 0
fresh_prod topk_topp_s2_c64_fresh --variant topk_topp --seed 2 --max-num-seqs 64 --rep 0
fresh_prod temp_s3_c64_fresh      --variant temp      --seed 3 --max-num-seqs 64 --rep 0
fresh_prod topk_topp_s3_c64_fresh --variant topk_topp --seed 3 --max-num-seqs 64 --rep 0
echo "[$(ts)] FRESH_PROD_DONE"
# ---- trace replay (vLLM enable_trace_replay + trace_decode_token_ids) with the logits hook, same split
for name in $NAMES; do fresh_replay $name; done
echo "[$(ts)] FRESH_REPLAY_DONE"
# ---- sampler.select over every generated position: independent (exact_k pinned) + same_runtime (vLLM dispatch)
for name in $NAMES; do analyze $name auto exact_k ""; done
for name in topk_topp_s1_c64_fresh topk_topp_s2_c64_fresh topk_topp_s3_c64_fresh; do
  analyze $name triton exact_k ""          # the Triton top-k/top-p kernel as the same_runtime reference
  analyze $name auto unpinned _unpinned    # abstaining checker: certified only where both tie rules agree
done
echo "[$(ts)] FRESH_ANALYZE_DONE"
# ---- negatives on the fresh records
for name in topk_topp_s1_c64_fresh temp_s1_c64_fresh; do
  python -m verity_vllm_sampler.negative n09 --production "$W3_RESULTS/prod/$name.json" --logits-dir "$W3_RESULTS/replay/logits_$name" \
      --out "$W3_RESULTS/negative/n09_$name.json" > "$W3_RESULTS/logs/n09_$name.log" 2>&1
  echo "[$(ts)] n09 $name rc=$? $(grep N09_DONE "$W3_RESULTS/logs/n09_$name.log" | cut -c1-300)"
done
python -m verity_vllm_sampler.negative n08 --production "$W3_RESULTS/prod/topk_topp_s1_c64_fresh.json" --honest-logits "$W3_RESULTS/replay/logits_topk_topp_s1_c64_fresh" \
    --index 6 --k 5 --out-dir "$W3_RESULTS/negative/n08_topk_topp_s1_fresh_i6_k5" > "$W3_RESULTS/logs/n08_topk_topp_fresh.log" 2>&1
echo "[$(ts)] n08 topk_topp rc=$? $(grep N08_DONE "$W3_RESULTS/logs/n08_topk_topp_fresh.log" | cut -c1-400)"
python -m verity_vllm_sampler.negative n08 --production "$W3_RESULTS/prod/greedy_c64_fresh.json" --honest-logits "$W3_RESULTS/replay/logits_greedy_c64_fresh" \
    --index 2 --k 3 --out-dir "$W3_RESULTS/negative/n08_greedy_fresh_i2_k3" > "$W3_RESULTS/logs/n08_greedy_fresh.log" 2>&1
echo "[$(ts)] n08 greedy rc=$? $(grep N08_DONE "$W3_RESULTS/logs/n08_greedy_fresh.log" | cut -c1-400)"
echo "FRESH_BATCH_DONE $(ts)"
