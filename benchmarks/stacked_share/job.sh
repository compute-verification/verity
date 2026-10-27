#!/usr/bin/env bash
# The exact-int8 Llama-3.1-8B request Program (LP=256, T=31, the row's shape), then units.py on it, as served and modeled at
# 70B TP8 (llama31-70b-tp8.json). Keeps the descriptor, the Build's result and both units documents in $OUT. The Build needs
# no GPU: it traces on the meta device at the explicit TARGET, and gives the row's own request Program (digest
# 8c683b2a...). DESCRIPTOR=PATH skips the Build and reads that one.
#
#   research run --queue --cpus 4 --mem-gb 32 --max-min 240 --source . --project verity \
#     --cwd clone/integrations/vllm -- bash ../../benchmarks/stacked_share/job.sh
set -euo pipefail
export PY=/workspace/jobs/venv312/bin/python3 PYTHONPATH=.:../..:../../tools/research/src HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false
ROW=llama31-8b__int8__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager
REPO=verity-int8/Llama-3.1-8B-INT8
REV=723d342ac443b040931242c768df342b0f6f15a7
TARGET='{"int8_exact": "all", "compute_capability": [12, 0], "num_sms": 188}'
HERE=../../benchmarks/stacked_share
if [ -z "${DESCRIPTOR:-}" ]; then
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
"$PY" -m verity_vllm.pipeline.cli workload --row-id "$ROW" --hf-repo "$REPO" --revision "$REV" --batch 1 --max-input 256 \
  --max-output 32 --case LLAMA31_8B_INT8 --target "$TARGET"
"$PY" -m verity_vllm.pipeline.cli build --model "$REPO" --revision "$REV" --wrapper request --LP 256 --T 31 --max-model-len 288 \
  --workload "workloads/$ROW.json" --out "$d/build_request" --target "$TARGET" > "$OUT/build_request.log" 2>&1 \
  || { tail -50 "$OUT/build_request.log"; exit 10; }
cp "$d/build_request/descriptor.json.gz" "$d/build_request/result.json" "$OUT/"
DESCRIPTOR=$OUT/descriptor.json.gz
fi
/usr/bin/time -v "$PY" "$HERE/units.py" "$DESCRIPTOR" --out "$OUT/units-8b.json" \
  --as "$HERE/llama31-70b-tp8.json" --as-out "$OUT/units-70b.json" 2> >(tee "$OUT/units.time" >&2)
