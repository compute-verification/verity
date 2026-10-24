#!/usr/bin/env bash
# The trace corpus (corpus.py's docstring) on one leased GPU of vy-nebius-2, from the shipped tree's root, with the C4 shard
# benchmarks/ppl/datasets.json pins (`research data fetch art:4f1d50db...`) sent along:
#   research run --on vy-nebius-2 --project verity --campaign glm-e2e-demo --source <tree> --cwd source \
#     --send c4-validation.00000-of-00008.json.gz [--env TRACES=heavy,light] [--env REQUESTS=1000] [--env MAX_MIN=45] -- \
#     bash benchmarks/network_e2e/traces/run.sh
# vLLM is the node's pinned venv (VENV, vLLM d9105ea80), as for benchmarks/network_traces: the workspace's environment has
# torch but not vLLM (integrations/vllm/pyproject.toml). MAX_MIN bounds the lease; the corpus stops 3 minutes before it.
# The corpus goes into the evidence store from here when the job can write to it, else custody uploads OUT.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; mkdir -p "$OUT"
VENV=${VENV:-/workspace/jobs/venv312}
MODEL=${MODEL:-/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7}
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
MAX_MIN=${MAX_MIN:-45}
[ -f "$C4" ] || { echo "trace-corpus: no C4 shard (--send it, or C4=PATH)"; exit 2; }
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.0f}
git rev-parse HEAD | tee "$OUT/tree.txt"
echo "trace-corpus: asking for one GPU at $(date -u +%H:%M:%SZ) for at most $MAX_MIN min"
gpu-lease 1 --wait --max-min "$MAX_MIN" --mem-gb 64 -- bash -c 'nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES%%,*}" > "$0/nvidia-smi.txt" 2>&1;
  echo "trace-corpus: GPU ${CUDA_VISIBLE_DEVICES} (${GPU_LEASE_UUID:-}) at $(date -u +%H:%M:%SZ)"; exec timeout "$1" "$2" "${@:3}"' \
  "$OUT" $(( (MAX_MIN - 3) * 60 )) "$VENV/bin/python" benchmarks/network_e2e/traces/corpus.py run --out "$OUT/corpus" \
  --model "$MODEL" --c4 "$C4" ${TRACES:+--traces "$TRACES"} --requests "${REQUESTS:-1000}"
rc=$?
echo "trace-corpus: corpus.py exited $rc at $(date -u +%H:%M:%SZ)"
"$VENV/bin/python" benchmarks/network_e2e/traces/corpus.py summarize "$OUT/corpus" || rc=${rc:-1}
rm -rf "$OUT/corpus/.site"
if research data put --kind vllm-trace-corpus/v1 --tree "$OUT/corpus" --preserve > "$OUT/data-put.txt" 2>&1; then
  echo "trace-corpus: data put: $(tail -1 "$OUT/data-put.txt")"
else
  echo "trace-corpus: data put from the job failed ($(tail -1 "$OUT/data-put.txt" | cut -c1-160)); custody uploads OUT"
fi
exit "$rc"
