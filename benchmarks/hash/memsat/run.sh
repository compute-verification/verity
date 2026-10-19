#!/usr/bin/env bash
# The hash loop's Commit + memory arm at saturation, workload llama8b-int8 (cwd = the source tree): memory accounting's
# benchmarks/remote_session/run.sh with SERVE=node as it is, but for its engines, which are engine_int8.py (engine.py at
# saturation on vllm's exact-int8 profile, with every request's token times), and its prover, which is node_paced.py
# (node.py with each session's passes at a rate of its own, NODE_ARGS' --pass-rates). run.sh and node.sh are copied into
# scratch with those substitutions, checked to have happened, so the recipe stays theirs. Every variable is run.sh's and
# node.sh's (ROLE, PORT, PLAN, PEER, SESSIONS, WINDOWS, WINDOW_MS, SAMPLES, ENGINE_ARGS, ENGINE_CPUS, NODE_ARGS, ...), except
# MODEL, which defaults to checkpoints-int8.json's LLAMA31_8B_INT8 pin under HF_HOME. After the prover, summarize.py
# writes $OUT/memsat.json and result.json (bench-result/v1).
#
#   research run --queue --on vy-nebius-2 --gpus 1 --gpu-ids 4 --preemptible --mem-gb 64 --disk-gb 1 --max-min 30 ... \
#     --env ROLE=prover --env PORT=7896 --env SERVE=node --env PLAN=honest,honest,honest --env WINDOW_MS=10000 \
#     --env ENGINE_CPUS=96-107 --env PROVER_CPUS=108-119 --env ENGINE_ARGS='--int8 linears --max-num-seqs 256
#     --concurrency 384 --step-hz 1000 --align-steps --arena-gib 9 --plain-s 100' --env NODE_ARGS='--decode-s 60
#     --after-s 40 --warm-openings 6713 --pass-rates 0,30,15' -- bash benchmarks/hash/memsat/run.sh
#
# The verifier is run.sh's own (ROLE=verifier on vy-nebius-1, as its north-star recipe), started first. Its registry keeps
# one live registration per prover node: a prover stopped mid-run (preempted) leaves its registration live, and the next
# is refused until `ROLE=verifier CLOSE=<node hex>` closes it. The engine's host work bounds plain's GPU share, so pick
# ENGINE_CPUS on the GPU's NUMA node where others' load is lowest at launch.
set -uo pipefail
M=benchmarks/hash/memsat
export HF_HOME=${HF_HOME:-/workspace/hf}
if [ -z "${MODEL:-}" ]; then
  MODEL=$(python3 - << 'EOF'
import json, os
pin = next(c for c in json.load(open("integrations/vllm/manifests/checkpoints-int8.json"))["checkpoints"]
           if c["role"] == "LLAMA31_8B_INT8")
print(f"{os.environ['HF_HOME']}/hub/models--{pin['repo'].replace('/', '--')}/snapshots/{pin['revision']}")
EOF
  )
fi
export MODEL
[ "${SERVE:-}" != node ] || [ -d "$MODEL" ] || { echo "no checkpoint at $MODEL"; exit 3; }
S=$(mktemp -d /tmp/memsat.XXXXXX); trap 'rm -rf "$S"' EXIT
sed -e "s#\$RS/engine.py#$M/engine_int8.py#" -e "s#\$RS/node.py#$M/node_paced.py#" benchmarks/remote_session/node.sh \
  > "$S/node.sh"
sed "s#bash \$RS/node.sh#bash $S/node.sh#" benchmarks/remote_session/run.sh > "$S/run.sh"
grep -q "$M/engine_int8.py" "$S/node.sh" && grep -q "$M/node_paced.py" "$S/node.sh" && grep -q "bash $S/node.sh" "$S/run.sh" \
  || { echo "run.sh or node.sh no longer runs \$RS/node.sh, \$RS/engine.py or \$RS/node.py: update $M/run.sh"; exit 3; }
bash "$S/run.sh"; P=$?
if [ "${ROLE:-both}" != verifier ] && [ -f "${RESEARCH_RUN_DIR:?}/out/prover.json" ]; then
  python3 $M/summarize.py "$RESEARCH_RUN_DIR/out" --result "$RESEARCH_RUN_DIR/result.json" || { [ $P = 0 ] && P=6; }
fi
exit $P
