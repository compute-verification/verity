#!/usr/bin/env bash
# aot.sh NEUTRAL_RUN ROW CASE WORKERS: the traced Build a neutral.sh run left on this node, declared as this run's vllm-build/v1
# output (`research-outputs write`), and compared with the predictor (`predict compare`) for `records switch`.
set -uo pipefail
NRUN=$1; ROW=$2; CASE=$3; WORKERS=${4:-8}
OUT="$RESEARCH_RUN_DIR/out"; mkdir -p "$OUT"
D=/workspace/tmp/circuits-neutrality/$NRUN/$ROW
git rev-parse HEAD > "$OUT/head.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/summary.txt"; }
say "row $ROW from $NRUN ($D) node $(hostname)"
[ -f "$D/build_summary.json" ] || { say "no build_summary.json under $D"; exit 3; }
uv sync -q --all-packages --extra torch-cpu > "$OUT/sync.log" 2>&1; say "sync rc=$?"
uv run -q --project tools/research research data fetch-fixtures --repo "$(pwd -P)" "integrations/vllm/workloads/$ROW.json" > "$OUT/fixtures.log" 2>&1
say "fixtures rc=$? $(sha256sum "integrations/vllm/workloads/$ROW.json" 2>&1 | cut -c1-16)"
cd integrations/vllm || exit 3
if [ "${SKIP_OUTPUTS:-0}" != 1 ]; then
  uv run -q --project ../.. python -m verity_vllm.pipeline.cli research-outputs write "$D" "$ROW" build --run-dir "$RESEARCH_RUN_DIR" \
    > "$OUT/outputs.log" 2>&1
  say "outputs rc=$?"
  cp "$RESEARCH_RUN_DIR/outputs.json" "$OUT/outputs.json" 2>/dev/null
  du -sh "$RESEARCH_RUN_DIR/outputs" >> "$OUT/summary.txt" 2>&1
fi
{ time uv run -q --project ../.. python -m verity_vllm.pipeline.cli predict compare "$D" --row "$ROW" \
    --config "verity_vllm/engine/profiles/hf_configs/$CASE.config.json" --out "$OUT/comparison.json" --workers "$WORKERS"; } \
  > "$OUT/compare.log" 2>&1
say "compare rc=$?"
python3 - "$OUT/comparison.json" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys
d = json.load(open(sys.argv[1]))
print("equal", d["equal"], "code_version", d.get("code_version"))
for r, p in sorted(d["ranks"].items(), key=lambda x: int(x[0])):
    bad = [n for n, v in p.items() if isinstance(v, dict) and "equal" in v and not v["equal"]]
    print("rank", r, "step", (p.get("step") or {}).get("traced", "")[:16], "parts", sum(1 for v in p.values() if isinstance(v, dict) and "equal" in v),
          "unequal", bad[:6], "errors", str(p.get("errors", ""))[:300])
EOF
tail -3 "$OUT/compare.log" >> "$OUT/summary.txt"
