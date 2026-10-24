#!/usr/bin/env bash
# The trace corpus (corpus.py's docstring) on one leased GPU of a node, from the shipped tree's root, with the C4 shard
# benchmarks/ppl/datasets.json pins (`research data fetch art:4f1d50db...`) sent along:
#   research run --on vy-nebius-1 --project verity --campaign glm-e2e-demo --source <tree> --cwd source \
#     --send c4-validation.00000-of-00008.json.gz [--env CORPORA="llama32_1b/seeded llama32_1b/async-greedy:heavy,smallkv"] \
#     [--env REQUESTS=1000] [--env MAX_MIN=45] -- bash benchmarks/network_e2e/traces/run.sh
# A CORPORA entry is MODEL/VARIANT[:TRACE,...] (corpus.MODELS, corpus.VARIANTS, corpus.TRACES; default llama32_1b/greedy,
# every trace). vLLM is the node's pinned venv (VENV, vLLM d9105ea80), as for benchmarks/network_traces: the workspace's
# environment has torch but not vLLM (integrations/vllm/pyproject.toml). MAX_MIN bounds the lease; each corpus runs in
# turn, into OUT/corpus/<model>-<variant>/, while more than a minute of it is left, and all stop 2 minutes before it ends.
# After the lease each corpus is summarized, and OUT/compare/<sampling>/ compares the output ids of the corpora that sampled
# alike (a variant and its async- twin).
# The node has no store credentials: custody uploads OUT, and `research data put --kind vllm-trace-corpus/v1 --tree
# OUT/corpus/<model>-<variant> --preserve` runs where `research fetch RUN --all` brought it.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
VENV=${VENV:-/workspace/jobs/venv312}
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.0f}
CORPORA=${CORPORA:-llama32_1b/greedy}
MAX_MIN=${MAX_MIN:-45}
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
[ -f "$C4" ] || { echo "trace-corpus: no C4 shard (--send it, or C4=PATH)"; exit 2; }

if [ -n "${TRACE_CORPUS_END:-}" ]; then                     # inside the lease
  nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES%%,*}" > "$OUT/nvidia-smi.txt" 2>&1
  echo "trace-corpus: GPU ${CUDA_VISIBLE_DEVICES} (${GPU_LEASE_UUID:-}) at $(date -u +%H:%M:%SZ)"
  rc=0
  for spec in $CORPORA; do
    mv=${spec%%:*}; traces=${spec#"$mv"}; traces=${traces#:}
    m=${mv%%/*}; v=${mv#*/}
    left=$(( TRACE_CORPUS_END - $(date +%s) ))
    if [ "$left" -le 60 ]; then echo "trace-corpus: $spec dropped: ${left} s of the lease left"; rc=1; continue; fi
    echo "trace-corpus: $spec at $(date -u +%H:%M:%SZ), at most ${left} s"
    timeout "$left" "$VENV/bin/python" benchmarks/network_e2e/traces/corpus.py run --model-id "$m" --variant "$v" \
      --out "$OUT/corpus/$m-$v" --c4 "$C4" ${traces:+--traces "$traces"} --requests "${REQUESTS:-1000}" || rc=$?
    echo "trace-corpus: $spec ended (rc so far $rc) at $(date -u +%H:%M:%SZ)"
  done
  exit "$rc"
fi

mkdir -p "$OUT"
git rev-parse HEAD | tee "$OUT/tree.txt"
echo "trace-corpus: asking for one GPU at $(date -u +%H:%M:%SZ) for at most $MAX_MIN min ($CORPORA)"
gpu-lease 1 --wait --max-min "$MAX_MIN" --mem-gb 64 -- \
  bash -c 'export TRACE_CORPUS_END=$(( $(date +%s) + ($1 - 2) * 60 )); exec bash "$0"' "$0" "$MAX_MIN"
rc=$?
echo "trace-corpus: lease over at $(date -u +%H:%M:%SZ)"
declare -A made                                             # sampling -> the corpora made with it
for spec in $CORPORA; do
  mv=${spec%%:*}; v=${mv#*/}; dir=$OUT/corpus/${mv%%/*}-$v
  [ -f "$dir/meta.json" ] || continue
  "$VENV/bin/python" benchmarks/network_e2e/traces/corpus.py summarize "$dir" || rc=1
  rm -rf "$dir/.site"
  made[${v#async-}]+="$dir "
done
for s in "${!made[@]}"; do
  set -- ${made[$s]}
  [ "$#" -gt 1 ] || continue
  "$VENV/bin/python" benchmarks/network_e2e/traces/corpus.py compare --out "$OUT/compare/$s" "$@" || rc=1
done
echo "trace-corpus: done (rc $rc) at $(date -u +%H:%M:%SZ)"
exit "$rc"
