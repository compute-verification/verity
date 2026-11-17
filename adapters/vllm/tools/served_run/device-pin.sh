#!/usr/bin/env bash
# This node's cuBLAS device pin, probed on its card under its lease (`run.sh device-pin`): out/device_pin.json.
set -euo pipefail
for v in CARD CORES VENV; do
  [ -n "${!v:-}" ] || { echo "device-pin.sh: no $v: start it with \`adapters/vllm/tools/served_run/run.sh device-pin\`"; exit 2; }
done
OUT=${RESEARCH_RUN_DIR:?run under research run}/out
mkdir -p "$OUT"
TREE=$PWD
cd adapters/vllm
PYTHONPATH=$TREE:$TREE/adapters/vllm:$TREE/adapters/torch:$TREE/tools/config:$TREE/tools/research/src \
  gpu-lease 1 --on "$CARD" --cores "$CORES" --max-min 30 --mem-gb 32 -- \
  "$VENV/bin/python" -m verity_vllm.pipeline.cli device-pin probe ${LIKE:+--like "$LIKE"} --out "$OUT/device_pin.json" \
  --evidence "${RESEARCH_RUN_ID:?}"
