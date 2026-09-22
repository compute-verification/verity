#!/bin/bash
# After run_fresh_batch.sh: (a) measure num_cached_tokens for every production-record structure that lacks it
# (re-run of the identical command with the witness-recording recorder), (b) re-run sampler.select on the
# cross-batch dumps with the correct production file and the pinned tie rule, (c) label all records,
# (d) compare.py + report.py.
#   W3_VENV=/workspace/venv-nightly W2_HARNESS=/workspace/w2harness/vllm-poc setsid nohup bash verity_vllm_sampler/pod/run_followup.sh > $W3_RESULTS/logs/followup.log 2>&1 < /dev/null &
set -uo pipefail
source "$(dirname "$0")/env.sh"
W2_HARNESS=${W2_HARNESS:-/workspace/w2harness/vllm-poc}
WHEEL_SHA=${WHEEL_SHA:-0198513ebd5561f154d9a5a241cdaacd1a7a211bff66aa8997e9e62481eb9120}
ADAPTER_REV=${ADAPTER_REV:-unknown}
POD_HOURS=${POD_HOURS:-}
ts() { date -u +%H:%M:%SZ; }
mkdir -p "$W3_RESULTS/diag"

while ! grep -q FRESH_BATCH_DONE "$W3_RESULTS/logs/fresh_batch.log" 2>/dev/null; do sleep 20; done
echo "[$(ts)] fresh batch done; measuring cached prefixes per structure"

cc() {  # name args...   (one GPU job at a time; fresh process; identical command to the original record)
  local name=$1; shift
  local out=$W3_RESULTS/diag/${name}_cachecheck.json
  if [ -s "$out" ]; then echo "[skip] cachecheck $name"; return; fi
  echo "[$(ts)] cachecheck $name"
  VLLM_ENABLE_V1_MULTIPROCESSING=1 python -m verity_vllm_sampler.production --workload "$W3_WORKLOAD" --out "$out" "$@" \
      > "$W3_RESULTS/logs/cachecheck_$name.log" 2>&1
  echo "[$(ts)] done  rc=$? $(grep PRODUCTION_DONE "$W3_RESULTS/logs/cachecheck_$name.log" | cut -c1-260)"
}
cc temp_s1_c64_r0          --variant temp      --seed 1 --max-num-seqs 64 --rep 0
cc topk_topp_s1_c64_r0     --variant topk_topp --seed 1 --max-num-seqs 64 --rep 0
cc greedy_c1_r0            --variant greedy             --max-num-seqs 1  --rep 0
cc greedy_c8_reversed      --variant greedy             --max-num-seqs 8  --rep 0 --order reversed
cc topk_topp_s1_c8_shuffled --variant topk_topp --seed 1 --max-num-seqs 8 --rep 0 --order shuffled:7
cc probes_greedy           --variant probes             --max-num-seqs 64 --probe-max-tokens 32
cc greedy_c8_r0            --variant greedy             --max-num-seqs 8  --rep 0
cc greedy_c32_r0           --variant greedy             --max-num-seqs 32 --rep 0
echo "[$(ts)] CACHECHECK_DONE"

# cross-batch dumps: sampler.select with the pinned tie rule and the right production file (the earlier
# reanalysis of these four failed on a wrong path and left the unpinned results in place)
export VLLM_ENABLE_V1_MULTIPROCESSING=0
for name in topk_topp_s1_c64_r0_at_c8 greedy_c64_r0_at_c8 temp_s1_c64_r0_at_c8 topk_topp_s1_c64_r0_at_c1; do
  prod=$W3_RESULTS/prod/${name%_at_c*}.json
  out=$W3_RESULTS/analysis/select_${name}_auto.json
  echo "[$(ts)] analyze $name"
  python -m verity_vllm_sampler.analyze --production "$prod" --logits-dir "$W3_RESULTS/replay/logits_$name" --out "$out" \
      --same-runtime --topk-topp-impl auto --topk-ties exact_k > "$W3_RESULTS/logs/analyze_${name}_auto.log" 2>&1
  echo "[$(ts)] done  rc=$? $(grep ANALYZE_DONE "$W3_RESULTS/logs/analyze_${name}_auto.log" | cut -c1-360)"
done

echo "[$(ts)] labelling"
python -m verity_vllm_sampler.fresh label-all --prod-dir "$W3_RESULTS/prod" --measured-dir "$W3_RESULTS/diag" --in-place \
    --out "$W3_RESULTS/cached_prefix_labels.json" 2>&1 | tail -3
python -m verity_vllm_sampler.compare --prod-dir "$W3_RESULTS/prod" --out "$W3_RESULTS/reproducibility.json" 2>&1 | tail -8
echo "[$(ts)] report"
PYTHONPATH=$W2_HARNESS:$PYTHONPATH python -m verity_vllm_sampler.report --raw "$W3_RESULTS" --out-root "$W3_RESULTS/tables" \
    --wheel-sha "$WHEEL_SHA" --lock /workspace/veritor/vllm-poc/env/requirements.sampler-nightly.lock \
    --adapter-revision "$ADAPTER_REV" ${POD_HOURS:+--pod-hours $POD_HOURS} 2>&1 | tail -5
echo "FOLLOWUP_DONE $(ts)"
