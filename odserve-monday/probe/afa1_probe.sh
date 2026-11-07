#!/usr/bin/env bash
# At the shipped tree on vy-dc-h200-2, CPU only: H1's window Program digests (program_probe.sh), the row's Build of record
# against the traced row (predict row, predict compare), and where Monday's settings file sits; prints no count.
set -uo pipefail
IN=$RESEARCH_RUN_DIR/inputs
echo "TREE $(git rev-parse HEAD)"
bash "$IN/program_probe.sh"
PY=/workspace/cp/venv312/bin/python NAME=llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager
ROW=/workspace/research/vllm-composed-h200int8/r20261102-233856-d4d6/$NAME T=$(mktemp -d)
export PYTHONPATH=$PWD:$PWD/integrations/vllm:$PWD/tools/research/src CUDA_VISIBLE_DEVICES= HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
"$PY" -m verity_vllm.pipeline.cli predict row --row "$NAME" --out "$T/pred" > "$T/row.json" 2> "$T/row.err"
echo "PREDICT-ROW exit $?: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("manifest_digest","")[:16], d.get("program_digest","")[:16])' "$T/row.json" 2>&1)"
CONFIG=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["config"])' "$T/pred/predicted.json")
"$PY" -m verity_vllm.pipeline.cli predict compare "$ROW" --row "$NAME" --config "$CONFIG" --predicted "$T/pred" \
  --out "$T/compare.json" > "$T/compare.log" 2>&1
echo "PREDICT-COMPARE against r20261102-233856-d4d6 exit $?"
find /workspace/research -maxdepth 5 -name 'settings-monday.env' 2>/dev/null | while read -r f; do echo "SETTINGS $f $(sha256sum < "$f" | cut -c1-16)"; done
rm -rf "$T"
