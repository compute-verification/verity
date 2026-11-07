#!/usr/bin/env bash
# The B4 H200 int8 row's Build of record at this tree (CPU only, target-declared, as the traced row was built), into
# /workspace/research/vllm-composed-h200int8/$RESEARCH_RUN_ID/ROW, then the harness's own B step against it: `predict row`
# at this tree into $RESEARCH_RUN_ID/predicted (or PRED, argv 1, if given) and `predict compare`. $OUT gets the build's
# logs, the manifest's digest and family counts and the compare's log, never the manifest.
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone -- bash retrace-row.sh [PRED]
set -uo pipefail
OUT=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$OUT"
ROW=llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager
ROLE=LLAMA32_1B_INT8 REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
TREE=$PWD ROOT=/workspace/research/vllm-composed-h200int8
BASE=$ROOT/$RESEARCH_RUN_ID D=$ROOT/$RESEARCH_RUN_ID/$ROW
PRED=${1:-$BASE/predicted}
mkdir -p "$BASE"
export PY=/workspace/cp/venv312/bin/python; export PY312=$PY HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PATH=/workspace/cp/venv312/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions
vv() { "$PY" -m verity_vllm.pipeline.cli "$@"; }
git -C "$TREE" rev-parse HEAD 2>/dev/null | tee "$OUT/tree.txt"
"$PY" -m research keep "$ROOT" --owner @vllm --expires 7d --priority high \
  --why "the H200 int8 Llama-3.2-1B B4 row's Build of record, served_verdicts.sh's predict-compare reference" 2>&1 | tail -1
(cd "$TREE/integrations/vllm" && vv row run "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$BASE" \
   --stages build --config-run 1 --replay-k 460) > "$OUT/build.out" 2>&1
rc=$?
for f in stages.txt row.log build_summary.json manifest.log; do cp "$D/$f" "$OUT/" 2>/dev/null; done
echo "ROW-BUILD $D rc=$rc"
[ "$rc" = 0 ] && [ -f "$D/manifest.json" ] || { tail -20 "$OUT/build.out"; exit 10; }
"$PY" - "$D/manifest.json" <<'PY' | tee "$OUT/manifest_summary.txt"
import collections, json, sys
m = json.load(open(sys.argv[1]))
print("ROW-MANIFEST", m["manifest_digest"], len(m["identities"]), dict(sorted(collections.Counter(r["family"] for r in m["identities"]).items())))
PY
if [ -z "${1:-}" ]; then
  vv predict row --row "$ROW" --out "$PRED" > "$OUT/predict_row.json" 2> "$OUT/predict_row.err"
  prc=$?
  [ "$prc" = 0 ] && [ -f "$PRED/predicted.json" ] || { echo "PREDICT-ROW refused (exit $prc): $(tail -3 "$OUT/predict_row.err" | cut -c1-600)"; exit 12; }
  echo "PREDICT-ROW $ROW: $(cut -c1-600 "$OUT/predict_row.json")"
fi
CONFIG=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["config"])' "$PRED/predicted.json")
vv predict compare "$D" --row "$ROW" --config "$CONFIG" --predicted "$PRED" \
  --out "$OUT/predict_compare.json" > "$OUT/predict_compare.log" 2>&1
crc=$?
echo "PREDICT-COMPARE against $D (exit $crc): $(grep -E '^(rank|predicted|\[predict\])' "$OUT/predict_compare.log" | tr '\n' ' ' | cut -c1-1200)"
exit $(( crc != 0 ? 11 : 0 ))
