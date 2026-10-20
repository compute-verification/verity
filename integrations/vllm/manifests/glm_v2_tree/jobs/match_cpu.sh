#!/usr/bin/env bash
# A v2 row's Match CPU phase (MATCH_PHASE=cpu: the tp2 check, the per-rank fold Match, compare, GM-01, the cross-rank leg) on a CPU
# node, over the Build of record and the GPU legs, both in the row dir: the legs copied from eu and checked against match_gpu.sh's
# match.sha256 (inputs/match.sha256, README "GPU batch"), all but its `<row>.tree` line: copy_rowdir.sh settled the row's tree, and
# tree_guard checks this job against it.  No GPU.
#   research run --on vy-glm53-cpu-2 --mem-gb 450 --mem-wait 28800 --timeout 20h --custody-ttl 24h --cwd source --project verity --send match.sha256 -- env ROW=... BUILD_RUN=... bash integrations/vllm/manifests/glm_v2_tree/jobs/match_cpu.sh
# env: ROW, BUILD_RUN, [SW=/workspace/cp/circuits/glm-v2/rows], [REF=cursor/glm-v2-21f3], [FOLD_RANK_WORKERS=3], [BUDGET_GB=150].
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}"
REV=aca966e4e02791568aa6a4ced368624b3d897f42
REF=${REF:-cursor/glm-v2-21f3}
SW=${SW:-/workspace/cp/circuits/glm-v2/rows}
W0=/workspace/cp/circuits
TREE=$PWD
D=$SW/$ROW
[ -f "$D/build_summary.json" ] && [ -s "$D/manifest.json" ] || { echo "no Build of record (build_summary.json, manifest.json) in $D"; exit 3; }
[ ! -e "$D/match/tp2_match.json" ] || { echo "$D/match already ran its check"; exit 3; }
if [ -s "$RESEARCH_RUN_DIR/inputs/match.sha256" ]; then
  grep -vF "  $ROW.tree" "$RESEARCH_RUN_DIR/inputs/match.sha256" | (cd "$SW" && sha256sum --quiet -c -) ||
    { echo "the GPU legs in $D/match are not eu's"; exit 3; }
fi
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" || exit 2
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
ldconfig -p | grep -q 'libcuda\.so\.1 ' || export LD_LIBRARY_PATH=${CUDA_DRIVER_DIR:-$W0/cuda-driver}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
(cd "$TREE/integrations/vllm" && WORK=$W0 VENV=$W0/venv312 HF_HOME=$W0/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$OUT/bootstrap") \
  >"$OUT/bootstrap.log" 2>&1 || { tail -30 "$OUT/bootstrap.log"; exit 5; }
export PY=$W0/venv312/bin/python
export PATH=$W0/venv312/bin:$PATH
export MATCH_PHASE=cpu FOLD_RANK_WORKERS=${FOLD_RANK_WORKERS:-3} HF_HOME=$W0/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
W=$(mktemp -d /tmp/glm-v2-mcpu.XXXXXX); trap 'rm -rf "$W"' EXIT
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$W/research" && chmod +x "$W/research"
export RESEARCH_BIN=$W/research TMPDIR=$W
"$RESEARCH_BIN" keep "$SW" --owner @circuits --expires 5d --priority normal \
  --why "GLM-5.3 v2 rows (cursor/glm-v2-21f3): Build of record and Match CPU phase ($RESEARCH_RUN_ID)" || echo "research keep failed (continuing)"
$PY -m verity_vllm.pipeline.cli row chain --stages match --build-run "$BUILD_RUN" --no-publish --ref "$REF" --by circuits \
  "$ROW" GLM53 zai-org/GLM-5.3 "$REV" --sweep-dir "$SW" --build-ram-budget-gb "${BUDGET_GB:-150}" >"$OUT/chain.out" 2>&1
rc=$?
for f in row.json stages.txt row.log match/tp2_match.json; do cp "$D/$f" "$OUT/" 2>/dev/null; done
tail -30 "$OUT/chain.out"
grep -E '^match ' "$D/stages.txt" | tail -1
exit $rc
