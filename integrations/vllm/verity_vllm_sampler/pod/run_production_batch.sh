#!/bin/bash
# All stock production runs for W3, one fresh process each (contract §11).  Runs detached on the pod:
#   setsid nohup bash verity_vllm_sampler/pod/run_production_batch.sh > $W3_RESULTS/logs/batch.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
run() {  # name, args...
  local name=$1; shift
  local out=$W3_RESULTS/prod/$name.json
  if [ -s "$out" ]; then echo "[skip] $name"; return; fi
  echo "[$(date -u +%H:%M:%SZ)] start $name"
  python -m verity_vllm_sampler.production --workload "$W3_WORKLOAD" --out "$out" "$@" > "$W3_RESULTS/logs/$name.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] done  $name rc=$? $(grep PRODUCTION_DONE "$W3_RESULTS/logs/$name.log" | cut -c1-160)"
}

# Phase A0: first records of each stochastic seed (feeds the replay dumps early)
run topk_topp_s1_c64_r0 --variant topk_topp --seed 1 --max-num-seqs 64 --rep 0
run temp_s1_c64_r0      --variant temp      --seed 1 --max-num-seqs 64 --rep 0
run greedy_c64_r0       --variant greedy             --max-num-seqs 64 --rep 0
# Phase A: three fresh-process repetitions per seed
for rep in 1 2; do
  run greedy_c64_r$rep --variant greedy --max-num-seqs 64 --rep $rep
done
for seed in 1 2 3; do
  for rep in 0 1 2; do
    run temp_s${seed}_c64_r$rep      --variant temp      --seed $seed --max-num-seqs 64 --rep $rep
    run topk_topp_s${seed}_c64_r$rep --variant topk_topp --seed $seed --max-num-seqs 64 --rep $rep
  done
done
# Phase B: concurrency caps {1, 8, 32} (64 above) + request orderings at cap 8
for cap in 1 8 32; do
  run greedy_c${cap}_r0       --variant greedy             --max-num-seqs $cap --rep 0
  run topk_topp_s1_c${cap}_r0 --variant topk_topp --seed 1 --max-num-seqs $cap --rep 0
  run temp_s1_c${cap}_r0      --variant temp      --seed 1 --max-num-seqs $cap --rep 0
done
run topk_topp_s1_c8_reversed --variant topk_topp --seed 1 --max-num-seqs 8 --rep 0 --order reversed
run topk_topp_s1_c8_shuffled --variant topk_topp --seed 1 --max-num-seqs 8 --rep 0 --order shuffled:7
run greedy_c8_reversed       --variant greedy             --max-num-seqs 8 --rep 0 --order reversed
# Phase D: stopping semantics
run probes_greedy        --variant probes           --max-num-seqs 64 --probe-max-tokens 32
run probes_temp_s1       --variant probes --seed 1  --max-num-seqs 64 --probe-max-tokens 32
run probes_greedy_stopnl --variant probes           --max-num-seqs 64 --probe-max-tokens 32 --stop-token-ids 198
run abort_greedy_15_20   --variant greedy           --max-num-seqs 64 --abort 15:20
run abort_topk_topp_s1_14_10 --variant topk_topp --seed 1 --max-num-seqs 64 --abort 14:10
# Phase C: Table 4 production overhead (stock = no logprobs, nothing kept), run alone on the GPU
run greedy_c64_norecord       --variant greedy             --max-num-seqs 64 --no-record
run topk_topp_s1_c64_norecord --variant topk_topp --seed 1 --max-num-seqs 64 --no-record
run greedy_c64_r0_again       --variant greedy             --max-num-seqs 64 --rep 3
echo "BATCH_DONE $(date -u +%H:%M:%SZ)"
