#!/usr/bin/env bash
# A v2 row's CPU tail: the deferred Commit's C2 replay of every rank (`tp-commit-replay`) from the bundles commit.sh preserved, then
# `records publish --tier heavy` citing each conformance property's capture of record by id (no --newest-captures: a job's store
# has no index), the Commit's run record and the Match's record (its three tp2 files stored as glm53/tp8-match-record/v1, as the v1
# records' were).  Runs on a CPU node holding the verified checkpoint (vy-nebius-cpu-3), with the Build of record copied into the row
# dir (the replay reads each rank's Build and the required manifest at the Commit's paths, README "GPU batch").  `--cwd clone`:
# publish writes manifests/records.json, kept in $OUT for the re-pin PR.  A rank's replay is ~70 min at ~265 GB peak (B8 greedy,
# r20261019-123645-e69d), so a row's ranks split over concurrent jobs on one node, RANKS=0,1,2,3 and RANKS=4,5,6,7 (each fetches
# its ranks' bundles, replays them, and exits 0 while ranks remain); the job without RANKS then folds the Commit if no RANKS job
# did, and publishes.
#   research run --on vy-nebius-cpu-3 --mem-gb 280 --mem-wait 28800 --timeout 16h --custody-ttl 24h --cwd clone --project verity --send bundles.json -- env ROW=... BUILD_ART=art:... MATCH_RUN=... COMMIT_RUN=... BF16_ART=art:... bash integrations/vllm/manifests/glm_v2_tree/jobs/replay_publish.sh
# env: ROW, BUILD_ART, MATCH_RUN, COMMIT_RUN, BF16_ART (cap_bf16_v6.sh's capture),
#      [RANKS], [SW=/workspace/cp/circuits/glm-v2/rows], [REF=cursor/glm-v2-21f3], [CK], [REPLAY_WORKERS=24].
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_ART:?BUILD_ART}" "${MATCH_RUN:?MATCH_RUN}" "${COMMIT_RUN:?COMMIT_RUN}" "${BF16_ART:?BF16_ART}"
REV=aca966e4e02791568aa6a4ced368624b3d897f42
REF=${REF:-cursor/glm-v2-21f3}
SW=${SW:-/workspace/cp/circuits/glm-v2/rows}
CK=${CK:-/workspace/hf/hub/models--zai-org--GLM-5.3/snapshots/$REV}
W0=/workspace/cp/circuits
TREE=$PWD
D=$SW/$ROW
B=$RESEARCH_RUN_DIR/inputs/bundles.json
[ -s "$B" ] || { echo "no inputs/bundles.json (commit.sh's \$OUT/bundles.json): --send it"; exit 2; }
[ -e "$CK/.verified" ] || { echo "no verified checkpoint at $CK"; exit 2; }
[ -s "$D/manifest.json" ] && [ -d "$D/build" ] || { echo "no Build of record in $D: copy it in first"; exit 2; }
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" || exit 2
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
(cd "$TREE/integrations/vllm" && WORK=$W0 VENV=$W0/venv312 HF_HOME=$W0/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$OUT/bootstrap") \
  >"$OUT/bootstrap.log" 2>&1 || { tail -30 "$OUT/bootstrap.log"; exit 5; }
PY=$W0/venv312/bin/python
export PATH=$W0/venv312/bin:$PATH
ldconfig -p | grep -q 'libcuda\.so\.1 ' || export LD_LIBRARY_PATH=${CUDA_DRIVER_DIR:-$W0/cuda-driver}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES= HF_HOME=$W0/hf HF_HUB_OFFLINE=1
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
W=$(mktemp -d "$SW/.replay.XXXXXX"); trap 'rm -rf "$W"' EXIT
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$W/research" && chmod +x "$W/research"
RB=$W/research
"$RB" keep "$SW" --owner @circuits --expires 7d --priority high \
  --why "GLM-5.3 v2 rows (cursor/glm-v2-21f3): the Commit's replay and record ($RESEARCH_RUN_ID)" || echo "research keep failed (continuing)"
fetch_tree() {  # ART DEST: the stored tree at DEST (replaced)
  local t; t=$(mktemp -d "$W/f.XXXXXX")
  "$RB" data fetch "$1" --to "$t/x" >/dev/null || return 1
  rm -rf "$2" && mkdir -p "$(dirname "$2")" && mv "$t/x" "$2"
}
meta=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["_rowmeta"]["art"])' "$B")
(
  flock 9
  if [ -e "$D/.rowmeta-${meta#art:}" ]; then echo "rowmeta $meta: already in $D"; exit 0; fi
  fetch_tree "$meta" "$W/rowmeta" || { echo "rowmeta $meta: fetch failed"; exit 6; }
  cp -a "$W/rowmeta/." "$D/" && touch "$D/.rowmeta-${meta#art:}" && echo "rowmeta $meta: $(find "$W/rowmeta" -type f | wc -l) files into $D"
) 9>"$D/.rowmeta.lock" || exit 6
python3 -c 'import json,sys; [print(k, v["art"], v["manifest_sha256"]) for k, v in json.load(open(sys.argv[1])).items() if k != "_rowmeta"]' "$B" |
while read -r name art sha; do
  r=${name##*_rank}
  [ -z "${RANKS:-}" ] || [[ ",$RANKS," == *",$r,"* ]] || continue
  if [ -f "$D/commit/$name/MANIFEST.json" ] && [ "$(sha256sum "$D/commit/$name/MANIFEST.json" | cut -c1-64)" = "$sha" ]; then echo "$name: in place"; continue; fi
  fetch_tree "$art" "$D/commit/$name" || { echo "$name $art: fetch failed"; exit 6; }
  [ "$(sha256sum "$D/commit/$name/MANIFEST.json" | cut -c1-64)" = "$sha" ] || { echo "$name: MANIFEST.json is not the Commit's"; exit 6; }
  echo "$name: fetched $art"
done || exit 6
$PY -m verity_vllm.pipeline.cli tp-commit-replay "$D/commit" --checkpoint-dir "$CK" --checkpoints-manifest "$TREE/integrations/vllm/manifests/checkpoints.json" \
  --model zai-org/GLM-5.3 --revision "$REV" --replay-workers "${REPLAY_WORKERS:-24}" ${RANKS:+--ranks "$RANKS"} >"$OUT/replay.out" 2>&1
rc=$?
cp "$D/commit/summary.json" "$OUT/commit_summary.json" 2>/dev/null; cp "$D"/commit/sampled_replay_p*_rank*.json "$OUT/" 2>/dev/null
tail -30 "$OUT/replay.out"
if [ -n "${RANKS:-}" ]; then
  case $rc in 0|4) echo "ranks $RANKS replayed (tp-commit-replay rc $rc); the job without RANKS publishes"; exit 0 ;; esac
fi
[ "$rc" = 0 ] || { echo "tp-commit-replay rc=$rc (3 FAIL, 4 ranks remain, 2 refused)"; exit "$rc"; }
COMMIT_ART=$("$RB" data show "$COMMIT_RUN" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["outputs"]["run_record"])') ||
  { echo "the store has no run_record for $COMMIT_RUN"; exit 2; }
mkdir -p "$W/match" && cp -p "$D/match/tp2_match.json" "$D/match/fold_match/tp2_fold_match.json" "$D/match/fold_match/tp2_rank_match.json" "$W/match/" ||
  { echo "no passing Match's tp2_match.json, fold_match/tp2_fold_match.json and fold_match/tp2_rank_match.json in $D/match"; exit 2; }
MATCH_ART=$("$RB" data put --kind glm53/tp8-match-record/v1 --tree "$W/match" --meta "{\"row\": \"$ROW\", \"run\": \"$MATCH_RUN\"}" --preserve |
  grep -oE 'art:[0-9a-f]{64}' | tail -1)
[ -n "$MATCH_ART" ] || { echo "research data put of the Match record failed"; exit 6; }
echo "match record $MATCH_ART, commit record $COMMIT_ART" | tee "$OUT/evidence_arts.txt"
CONF=(
  --conformance deepgemm_capture=art:407dd8a4c2be83ed4f421c7603ba395b700099023a41c3d61f5a3ce3ee3f3481
  --conformance glm53_tp8_allreduce_capture=art:f248a6771ee6efdf2464743e0e35d60cd26391cbae1a9f8fb6aae287ee8bc4f5
  --conformance glm53_dsa_capture=art:9a0a67752100db139bd4f00d52ba594618a62d480e40a7472c2b56ed37885dfb
  --conformance glm53_block_fp8_capture=art:41943522418e851b5fbd10853637ea34c0f8115118e60f7e5dc36f4473de74b8
  --conformance sigmoid_router_capture=art:f0439f03ce0d14b1d1281871806042a4c68afaa7ab66a2ad5daab254b25cec27
  --conformance glm53_fused_rmsnorm_capture=art:e98ef16cd853c142ba5f8fa7f82335e8e26d73fcb7bae2ad3e5d2888d8394e74
  --conformance glm53_fa3_split_capture=art:4e04491a2ac8a3c1225957a230312d27e434a894c055dffc23e390d6b430efcf
  --conformance glm53_cuda_rmsnorm_capture=art:5e724c60ec44bede350bfa8ba0c94d6ad6f6179679e743de3653e62665b424c7
  --conformance "glm53_bf16_gemm_capture=$BF16_ART"
)
cd "$TREE/integrations/vllm"
$PY -m verity_vllm.pipeline.cli records publish "$D" "$ROW" --build "$BUILD_ART" --tree "$REF@${RESEARCH_SOURCE_SHA:-$(git -C "$TREE" rev-parse HEAD)}" --by circuits --tier heavy \
  --match "$MATCH_ART" --match-run "$MATCH_RUN" --commit "$COMMIT_ART" --commit-run "$COMMIT_RUN" "${CONF[@]}" 2>&1 | tee "$OUT/publish.out"
prc=${PIPESTATUS[0]}
git -C "$TREE" diff -- integrations/vllm/manifests/records.json >"$OUT/records_json.diff"
exit "$prc"
