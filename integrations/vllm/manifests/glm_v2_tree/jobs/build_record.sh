#!/usr/bin/env bash
# A v2 row's Build of record on a CPU node: the AOT Build (`--build-construction record`: each derive `predict emit`, refused unless
# its Program equals the `aot.trace` Build's) with its required-value manifest and call-boundary check, as a vllm.build Attempt
# (`row chain --stages build --no-publish`), in the sweep the Match CPU phase and the Commit read (README "GPU batch").  A row whose
# v2 trace moved a Program takes that trace as AOT_TRACE (trace_compare.sh's row dir) and is re-pinned with it.  Hashes the row dir
# into $OUT/rowdir.sha256 for the copies to eu and to the replay host.
#   research run --on vy-glm53-cpu-2 --mem-gb 450 --mem-wait 28800 --timeout 20h --custody-ttl 24h --cwd source --project verity -- env ROW=... bash integrations/vllm/manifests/glm_v2_tree/jobs/build_record.sh
# env: ROW, [AOT_TRACE], [SW=/workspace/cp/circuits/glm-v2/rows], [REF=cursor/glm-v2-21f3], [BUDGET_GB=400], [VERITY_MANIFEST_JOBS=2].
set -uo pipefail
: "${ROW:?ROW}"
REV=aca966e4e02791568aa6a4ced368624b3d897f42
REF=${REF:-cursor/glm-v2-21f3}
SW=${SW:-/workspace/cp/circuits/glm-v2/rows}
W0=/workspace/cp/circuits
TREE=$PWD
D=$SW/$ROW
[ ! -e "$D/build_summary.json" ] || { echo "$D already holds a Build"; exit 3; }
mkdir -p "$SW" || exit 3
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" first || exit 2
SNAP=$W0/hf/hub/models--zai-org--GLM-5.3/snapshots/$REV
[ -s "$SNAP/config.json" ] && [ -s "$SNAP/model.safetensors.index.json" ] || { echo "no GLM-5.3 metadata at $SNAP"; exit 3; }
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
ldconfig -p | grep -q 'libcuda\.so\.1 ' || export LD_LIBRARY_PATH=${CUDA_DRIVER_DIR:-$W0/cuda-driver}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
(cd "$TREE/integrations/vllm" && WORK=$W0 VENV=$W0/venv312 HF_HOME=$W0/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$OUT/bootstrap") \
  >"$OUT/bootstrap.log" 2>&1 || { tail -30 "$OUT/bootstrap.log"; exit 5; }
export PY=$W0/venv312/bin/python
export PATH=$W0/venv312/bin:$PATH HF_HOME=$W0/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
export VERITY_MANIFEST_JOBS=${VERITY_MANIFEST_JOBS:-2}
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
W=$(mktemp -d /workspace/research/scratch/glmv2-build.XXXXXX); trap 'rm -rf "$W"' EXIT
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$W/research" && chmod +x "$W/research"
export RESEARCH_BIN=$W/research TMPDIR=$W
"$RESEARCH_BIN" keep "$SW" --owner @circuits --expires 5d --priority normal \
  --why "GLM-5.3 v2 rows (cursor/glm-v2-21f3): Build of record ($RESEARCH_RUN_ID) for the Match CPU phase, the Commit and the replay" \
  || echo "research keep failed (continuing)"
cd "$TREE/integrations/vllm" || exit 6
/usr/bin/time -v "$PY" -m verity_vllm.pipeline.cli row chain --stages build --no-publish --ref "$REF" --by circuits "$ROW" GLM53 zai-org/GLM-5.3 "$REV" \
  --sweep-dir "$SW" --build-construction record ${AOT_TRACE:+--aot-trace "$AOT_TRACE"} --build-ram-budget-gb "${BUDGET_GB:-400}" >"$OUT/chain.out" 2>&1
rc=$?
for f in build_summary.json row.json stages.txt; do cp "$D/$f" "$OUT/" 2>/dev/null; done
grep -E '^build |Elapsed|Maximum resident' "$OUT/chain.out" | tail -4
grep -q '^build PASS' "$D/stages.txt" 2>/dev/null || { echo "the Build did not pass ($D/stages.txt)"; exit $(( rc ? rc : 7 )); }
(cd "$SW" && { echo "$ROW.tree"; find "$ROW" -type f -not -path "$ROW/match/*" -not -path "$ROW/commit*"; } | tr '\n' '\0' |
  xargs -0 -P 16 -n 256 sha256sum) | sort -k2 >"$OUT/rowdir.sha256"
echo "rowdir.sha256: $(wc -l <"$OUT/rowdir.sha256") files"
exit $rc
