#!/usr/bin/env bash
# The certified tick's load point (corpus.py's --admit tick) for GLM-5.3 at TP8 on a whole 8-GPU H200 node, from the shipped
# tree's root, as a job whose input is the C4 shard benchmarks/ppl/datasets.json pins (--send it, or C4=PATH):
#   RATE=R [REQUESTS=N] [TRACE=heavy] [VARIANT=greedy] [TICK_FROM=DIR] [MAX_MIN=180] bash benchmarks/network_e2e/traces/tick.sh
# One plan at RATE requests/s on TRACE's server (corpus.TRACES), served twice, each a corpus of its own: OUT/corpus/arrival,
# vLLM adding each request as it comes, then OUT/corpus/tick under the certified tick, τ_sat and τ_free calibrated on
# arrival's trace (corpus.py tick). TICK_FROM, a summarized trace of the same server on this node, serves the tick alone,
# calibrated on it. REQUESTS is 1,300 for each request/s of RATE and at least 1,000 unless given, so that the arrivals span
# about 20 minutes, two 600 s windows; a corpus of the same REQUESTS serves the same plan. After the GPUs are given back the
# tick's corpus is summarized (each summary.json's `serving` has the trace's throughput and latency), and
# benchmarks/network_e2e/encodings/v1.py charges the traces at T = 6000 with the checkpoint's stop ids into OUT/advice.json
# and its table into OUT/advice.md, where the tick's timing is 0 bits.
# The node's setup is the GLM-5.3 TP drivers' (benchmarks/served_commit/tp_decode.sh): CHECKPOINT (default the node's
# verified local copy, else /workspace/fs/glm53/REV), the Hub offline, and VENV (default /workspace/cp/venv312), whose vLLM
# must be d9105ea80, the scheduler the replay models. Under gpu-lease where the node has it: 8 GPUs, at most MAX_MIN minutes.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
RATE=${RATE:?RATE: the load point in requests/s}
TRACE=${TRACE:-heavy}; VARIANT=${VARIANT:-greedy}; MAX_MIN=${MAX_MIN:-180}
REQUESTS=${REQUESTS:-$(awk -v r="$RATE" 'BEGIN { n = int(1300 * r + 0.999); print (n < 1000 ? 1000 : n) }')}
REV=aca966e4e02791568aa6a4ced368624b3d897f42
LOCAL_CK=/workspace/cp/glm53-ckpt/$REV
if [ -e "$LOCAL_CK/.verified" ]; then CK=$LOCAL_CK; else CK=/workspace/fs/glm53/$REV; fi
CHECKPOINT=${CHECKPOINT:-$CK}
VENV=${VENV:-/workspace/cp/venv312}; PY=$VENV/bin/python
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
CORPUS=benchmarks/network_e2e/traces/corpus.py
FROM=${TICK_FROM:-$OUT/corpus/arrival/$TRACE}
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export HF_HOME=${HF_HOME:-/workspace/cp/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [ -n "${TICK_LEASED:-}" ]; then                          # holding the GPUs
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tee "$OUT/nvidia-smi.txt"
  args=(--model-id glm53 --variant "$VARIANT" --checkpoint "$CHECKPOINT" --c4 "$C4" --traces "$TRACE" --rate "$RATE"
        --requests "$REQUESTS" --startup-s 1800)
  if [ -z "${TICK_FROM:-}" ]; then
    echo "tick: arrival at $(date -u +%H:%M:%SZ)"
    "$PY" "$CORPUS" run "${args[@]}" --out "$OUT/corpus/arrival" && "$PY" "$CORPUS" summarize "$OUT/corpus/arrival" || exit 1
  fi
  echo "tick: the certified tick at $(date -u +%H:%M:%SZ), calibrated on $FROM"
  exec "$PY" "$CORPUS" run "${args[@]}" --admit tick --tick-from "$FROM" --out "$OUT/corpus/tick"
fi

[ -f "$C4" ] || { echo "tick: no C4 shard (--send it, or C4=PATH)"; exit 2; }
test -s "$CHECKPOINT/config.json" || { echo "tick: no checkpoint at $CHECKPOINT (CHECKPOINT=DIR)"; exit 2; }
v=$("$PY" -c 'import vllm; print(vllm.__version__)') || { echo "tick: no vLLM in $VENV (VENV=DIR)"; exit 2; }
[[ $v == *+gd9105ea8* ]] || { echo "tick: $VENV has vLLM $v, not d9105ea80, whose scheduler the replay models (VENV=DIR)"; exit 2; }
mkdir -p "$OUT"
git rev-parse HEAD | tee "$OUT/tree.txt"
echo "tick: GLM-5.3 ($CHECKPOINT), vLLM $v, $REQUESTS requests at $RATE/s on $TRACE's server, $VARIANT, at $(date -u +%H:%M:%SZ)"
if [ -z "${GPU_LEASE_UUID:-}" ] && command -v gpu-lease >/dev/null 2>&1; then
  gpu-lease 8 --wait --max-min "$MAX_MIN" -- env TICK_LEASED=1 bash "$0"
else
  TICK_LEASED=1 bash "$0"
fi
rc=$?
echo "tick: served (rc $rc) at $(date -u +%H:%M:%SZ)"
rm -rf "$OUT"/corpus/*/.site
[ -f "$OUT/corpus/tick/meta.json" ] || exit 1
"$PY" "$CORPUS" summarize "$OUT/corpus/tick" || rc=1
stops=$("$PY" -c 'import json, sys; print(",".join(map(str, json.load(open(sys.argv[1]))["eos_token_ids"])))' \
  "$OUT/corpus/tick/meta.json")
traces=("tick=$OUT/corpus/tick/$TRACE")
[ -f "$FROM/steps.jsonl" ] && traces=("arrival=$FROM" "${traces[@]}")
PYTHONPATH=$PWD "$PY" -P benchmarks/network_e2e/encodings/v1.py "${traces[@]}" --T 6000 --stops "$stops" \
  --out "$OUT/advice.json" | tee "$OUT/advice.md" || rc=1
echo "tick: done (rc $rc) at $(date -u +%H:%M:%SZ)"
exit "$rc"
