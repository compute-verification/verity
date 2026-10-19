#!/usr/bin/env bash
# circuits WS-A (V2PREP reuses it unchanged): `verity-vllm predict compare` on a CPU pod from the shipped tree (--cwd source),
# against a traced Build's build_summary.json and per-rank build_step/result.json sent as $TRACE_TGZ.  ROW, CONFIG (relative to
# integrations/vllm), OUT, [WORKERS].  On the GLM v2 tree with v1's traces: rank-by-rank equal means v2 moved no GLM Program.
set -uo pipefail
S=$(mktemp -d /workspace/research/scratch/aotcmp.XXXXXX) || exit 3
trap 'rm -rf "$S"' EXIT
uv venv -q -p 3.12 "$S/v" && uv pip install -q -p "$S/v/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu" \
  && uv pip install -q -p "$S/v/bin/python" "numpy==2.5.3" || exit 4
export PYTHONPATH=$PWD/integrations/vllm:$PWD
mkdir -p "$OUT" "$S/row"
tar xzf "$RESEARCH_RUN_DIR/inputs/$TRACE_TGZ" -C "$S/row" || exit 5
cp "$S/row/build_summary.json" "$OUT/traced_build_summary.json"
cd integrations/vllm || exit 6
/usr/bin/time -v "$S/v/bin/python" -m verity_vllm.pipeline.cli predict compare "$S/row" --row "$ROW" --config "$CONFIG" \
  --out "$OUT/comparison.json" ${WORKERS:+--workers "$WORKERS"} 2>"$OUT/compare.err"
rc=$?
tail -n 30 "$OUT/compare.err" | grep -v "^\s*$" | head -40
exit $rc
