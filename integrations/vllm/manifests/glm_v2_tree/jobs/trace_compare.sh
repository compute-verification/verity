#!/usr/bin/env bash
# circuits V2PREP: one GLM-5.3 row's Programs on this tree, traced on CPU and compared both ways:
#   1. `verity-vllm neutrality ROW --tier heavy` (`row run --stages build --build-construction trace --build-programs-only=1`):
#      the traced Build's per-rank Program digests against manifests/records.json's pins (did v2 move a Program?);
#   2. `verity-vllm predict compare` of this tree's predictor against that trace (would the row's AOT Build emit it?).
# The CPU venv and checkpoint metadata are the ones aot_build_rm.sh (WS-E) keeps on vy-glm53-cpu-2 under W.  The traced row dir
# stays on the node (kept 5d): a row whose pins moved takes it as its AOT Build's --aot-trace.
# env: ROW, [W (/workspace/cp/circuits)], [BUDGET_GB (100)], [WORKERS (2)].  cwd: the source root.
# outputs: neutrality.out, neutrality.json, comparison.json, compare.err, trace.tgz (build_summary.json and every
# build/rank*/build_*/result.json: what `predict compare` reads), rowdir.txt.
set -uo pipefail
: "${ROW:?ROW}"
REV=aca966e4e02791568aa6a4ced368624b3d897f42
TREE=$PWD
W=${W:-/workspace/cp/circuits}
SW=$W/glm-v2/$RESEARCH_RUN_ID
SNAP=$W/hf/hub/models--zai-org--GLM-5.3/snapshots/$REV
[ -s "$SNAP/config.json" ] && [ -s "$SNAP/model.safetensors.index.json" ] || { echo "no GLM-5.3 metadata at $SNAP"; exit 3; }
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
exit $nrc
