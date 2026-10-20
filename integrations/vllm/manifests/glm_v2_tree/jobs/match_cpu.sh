#!/usr/bin/env bash
# A v2 row's Match CPU phase (MATCH_PHASE=cpu: the tp2 check, the per-rank fold Match, compare, GM-01, the cross-rank leg) on a CPU
# node, over the Build of record and the GPU legs, both in the row dir: the legs copied from window A's node and checked against match_gpu.sh's
# match.sha256 (inputs/match.sha256, README "GPU batch"), all but its `<row>.tree` line: copy_rowdir.sh settled the row's tree, and
# tree_guard checks this job against it.  No GPU.  Every file the phase writes under the row dir (the check, the folds, a
# served-shape derive's Programs) is hashed into $OUT/match_cpu.sha256 (paths relative to $SW), the list copy_rowdir.sh takes
# to window B's node, which holds the legs already when it ran window A.
#   research run --on vy-glm53-cpu-2 --mem-gb 450 --mem-wait 28800 --timeout 30h --custody-ttl 24h --cwd source --project verity --send match.sha256 -- env ROW=... BUILD_RUN=... bash integrations/vllm/manifests/glm_v2_tree/jobs/match_cpu.sh
# env: ROW, BUILD_RUN, [SW], [REF] (model.sh's defaults), [FOLD_WORKERS=2], [FOLD_RANK_WORKERS: the ranks whose compare and GM-01
# legs run at once, as many as the run's --mem-gb holds at GM_GB a rank beside FOLD_WORKERS folds at 40 GB], [GM_GB: a GM-01 leg's
# peak, by batch: B8 370 (339 GiB, WSBSTOCH's r20261019-162751-9b01), B1 60 and B32 1500 scaled from it, unmeasured],
# [FOLD_TIMEOUT=86400 a leg: row_tp's default 7200 s bounds the whole fold Match at 8 h, and a B8 row's takes about 17 h one GM-01
# at a time], [BUDGET_GB=150].
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}" "${RESEARCH_RUN_MEM_GB:?research run --mem-gb sizes the fold Match}"
case $ROW in *__b1__*) gm=60 ;; *__b32__*) gm=1500 ;; *) gm=370 ;; esac
GM_GB=${GM_GB:-$gm} FOLD_WORKERS=${FOLD_WORKERS:-2} mem=${RESEARCH_RUN_MEM_GB%.*}
n=$(( (mem - 40 * FOLD_WORKERS) / GM_GB )); [ "$n" -le 8 ] || n=8
FOLD_RANK_WORKERS=${FOLD_RANK_WORKERS:-$n}
[ "$FOLD_RANK_WORKERS" -ge 1 ] || { echo "--mem-gb $mem holds no GM-01 leg of $GM_GB GB beside $FOLD_WORKERS folds of 40 GB"; exit 3; }
W0=/workspace/cp/circuits
TREE=$PWD
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/model.sh"
D=$SW/$ROW
[ -f "$D/build_summary.json" ] && [ -s "$D/manifest.json" ] || { echo "no Build of record (build_summary.json, manifest.json) in $D"; exit 3; }
[ ! -e "$D/match/tp2_match.json" ] || { echo "$D/match already ran its check"; exit 3; }
if [ -s "$RESEARCH_RUN_DIR/inputs/match.sha256" ]; then
  grep -vF "  $ROW.tree" "$RESEARCH_RUN_DIR/inputs/match.sha256" | (cd "$SW" && sha256sum --quiet -c -) ||
    { echo "the GPU legs in $D/match are not the ones match_gpu.sh hashed"; exit 3; }
fi
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" || exit 2
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
ldconfig -p | grep -q 'libcuda\.so\.1 ' || export LD_LIBRARY_PATH=${CUDA_DRIVER_DIR:-$W0/cuda-driver}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
(cd "$TREE/integrations/vllm" && WORK=$W0 VENV=$W0/venv312 HF_HOME=$W0/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$OUT/bootstrap") \
  >"$OUT/bootstrap.log" 2>&1 || { tail -30 "$OUT/bootstrap.log"; exit 5; }
export PY=$W0/venv312/bin/python
export PATH=$W0/venv312/bin:$PATH
export MATCH_PHASE=cpu FOLD_RANK_WORKERS FOLD_WORKERS FOLD_TIMEOUT=${FOLD_TIMEOUT:-86400} HF_HOME=$W0/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
echo "fold Match: $FOLD_RANK_WORKERS ranks' compare and GM-01 legs at once ($GM_GB GB each), $FOLD_WORKERS folds, $FOLD_TIMEOUT s a leg (--mem-gb $mem)"
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
W=$(mktemp -d /tmp/$V2-mcpu.XXXXXX); trap 'rm -rf "$W"' EXIT
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$W/research" && chmod +x "$W/research"
export RESEARCH_BIN=$W/research TMPDIR=$W
"$RESEARCH_BIN" keep "$SW" --owner @circuits --expires 5d --priority normal \
  --why "$WHAT v2 rows ($REF): Build of record and Match CPU phase ($RESEARCH_RUN_ID)" || echo "research keep failed (continuing)"
M0=$W/started && touch "$M0"
$PY -m verity_vllm.pipeline.cli row chain --stages match --build-run "$BUILD_RUN" --no-publish --ref "$REF" --by circuits \
  "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$SW" --build-ram-budget-gb "${BUDGET_GB:-150}" >"$OUT/chain.out" 2>&1
rc=$?
for f in row.json stages.txt row.log match/tp2_match.json; do cp "$D/$f" "$OUT/" 2>/dev/null; done
(cd "$SW" && find "$ROW" -type f -newer "$M0" -print0 | xargs -0 -r -P 16 -n 256 sha256sum) | sort -k2 >"$OUT/match_cpu.sha256"
echo "match_cpu.sha256: $(wc -l <"$OUT/match_cpu.sha256") files"
tail -30 "$OUT/chain.out"
grep -E '^match ' "$D/stages.txt" | tail -1
exit $rc
