#!/usr/bin/env bash
# circuits V2PREP: one GLM-5.3 row's Programs on this tree, traced on CPU, compared both ways, and its AOT Build rehearsed:
#   1. `verity-vllm neutrality ROW --tier heavy` (`row run --stages build --build-construction trace --build-programs-only=1`):
#      the traced Build's per-rank Program digests against manifests/records.json's pins (did v2 move a Program?);
#   2. `verity-vllm predict compare` of this tree's predictor against that trace (would the row's AOT Build emit it?);
#   3. the row's AOT Build (`row chain --stages build --no-publish`, construction record: each derive `predict emit` against this
#      trace, `--aot-trace`, refused where it is not the trace's Program), Programs only (`--build-programs-only=1`: no manifest),
#      as a vllm.build Attempt of its own; then its digests against the trace's (aot_vs_trace.json) and the pins (aot_neutrality.out).
# The CPU venv and checkpoint metadata are the ones aot_build_rm.sh (WS-E) keeps on vy-glm53-cpu-2 under W.  The traced row dir
# stays on the node (kept 5d): a row whose pins moved takes it as its AOT Build's --aot-trace.
# env: ROW, [W (/workspace/cp/circuits)], [BUDGET_GB (100)], [WORKERS (2)], [SKIP_AOT=1].  cwd: the source root.
# outputs: neutrality.out, neutrality.json, comparison.json, compare.err, trace.tgz (build_summary.json and every
# build/rank*/build_*/result.json: what `predict compare` reads), rowdir.txt, aot_chain.out, aot_build_summary.json, aot_row.json,
# aot_vs_trace.json, aot_neutrality.out.
set -uo pipefail
: "${ROW:?ROW}"
REV=aca966e4e02791568aa6a4ced368624b3d897f42
TREE=$PWD
W=${W:-/workspace/cp/circuits}
SW=$W/glm-v2/$RESEARCH_RUN_ID
SNAP=$W/hf/hub/models--zai-org--GLM-5.3/snapshots/$REV
[ -s "$SNAP/config.json" ] && [ -s "$SNAP/model.safetensors.index.json" ] || { echo "no GLM-5.3 metadata at $SNAP"; exit 3; }
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -n "${SKIP_AOT:-}" ] || [ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
mkdir -p "$SW" || exit 3
git -C "$TREE" rev-parse HEAD >"$OUT/tree.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" >"$OUT/tree.txt"
(cd integrations/vllm && WORK=$W VENV=$W/venv312 HF_HOME=$W/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$OUT/bootstrap") \
  >"$OUT/bootstrap.log" 2>&1 || { tail -30 "$OUT/bootstrap.log"; exit 5; }
tail -1 "$OUT/bootstrap.log"
export PY=$W/venv312/bin/python HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
export TMPDIR=$(mktemp -d /workspace/research/scratch/glmv2.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT
export PATH=$W/venv312/bin:$PATH
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
"$PY" -m research keep "$SW" --owner @circuits --expires 5d --priority normal \
  --why "GLM v2 traced Build of $ROW ($RESEARCH_RUN_ID): the AOT Build's --aot-trace if v2 moved its Programs" || echo "research keep failed (continuing)"
echo "$SW/$ROW" >"$OUT/rowdir.txt"
cd integrations/vllm || exit 6
/usr/bin/time -v "$PY" -m verity_vllm.pipeline.cli neutrality "$ROW" --tier heavy --out "$SW" \
  --row-flag=--build-ram-budget-gb=${BUDGET_GB:-100} >"$OUT/neutrality.out" 2>&1
nrc=$?
tail -40 "$OUT/neutrality.out"
cp "$SW/neutrality.json" "$OUT/" 2>/dev/null
D=$SW/$ROW
for f in stages.txt row.log build_summary.json; do cp "$D/$f" "$OUT/" 2>/dev/null; done
[ -s "$D/build_summary.json" ] || { echo "no build_summary.json: the trace did not finish (neutrality rc $nrc)"; exit 7; }
(cd "$D" && tar czf "$OUT/trace.tgz" build_summary.json $(ls -d build/rank*/build_*/result.json 2>/dev/null))
/usr/bin/time -v "$PY" -m verity_vllm.pipeline.cli predict compare "$D" --row "$ROW" \
  --config verity_vllm/engine/profiles/hf_configs/GLM53.config.json --out "$OUT/comparison.json" --workers "${WORKERS:-2}" 2>"$OUT/compare.err"
crc=$?
grep -E "Elapsed|Maximum resident" "$OUT/compare.err"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("predict compare equal:", d.get("equal"))' "$OUT/comparison.json" 2>/dev/null
echo "neutrality rc $nrc, predict compare rc $crc"
[ "$crc" = 0 ] || exit 8
[ -n "${SKIP_AOT:-}" ] && exit $nrc
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$SW/research" && chmod +x "$SW/research"
export RESEARCH_BIN=$SW/research
A=$SW/aot
/usr/bin/time -v "$PY" -m verity_vllm.pipeline.cli row chain --stages build --no-publish --by circuits "$ROW" GLM53 zai-org/GLM-5.3 "$REV" \
  --sweep-dir "$A" --build-construction record --aot-trace "$D" --build-programs-only=1 --build-ram-budget-gb "${BUDGET_GB:-100}" \
  >"$OUT/aot_chain.out" 2>&1
arc=$?
tail -20 "$OUT/aot_chain.out"
cp "$A/$ROW/build_summary.json" "$OUT/aot_build_summary.json" 2>/dev/null
cp "$A/$ROW/row.json" "$OUT/aot_row.json" 2>/dev/null
"$PY" - "$D" "$A/$ROW" "$OUT/aot_vs_trace.json" <<'EOF'
import json, sys
from pathlib import Path
from verity_vllm.pipeline import epoch, neutrality
t, a = (epoch.row_digests(Path(p))["programs"] for p in sys.argv[1:3])
diffs = neutrality.compare(t, a) if t and a else ["a Build has no build_summary.json"]
Path(sys.argv[3]).write_text(json.dumps({"trace": sys.argv[1], "aot": sys.argv[2], "equal": not diffs, "differences": diffs,
                                         "programs": a}, indent=1) + "\n")
print("AOT vs trace:", "EQUAL" if not diffs else "DIFFER", *diffs, sep="\n    ")
EOF
"$PY" -m verity_vllm.pipeline.cli neutrality "$ROW" --tier heavy --compare --out "$A" >"$OUT/aot_neutrality.out" 2>&1
tail -12 "$OUT/aot_neutrality.out"
echo "neutrality rc $nrc, predict compare rc $crc, AOT Build rc $arc"
[ "$arc" = 0 ] || exit 9
exit $nrc
