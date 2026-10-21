#!/usr/bin/env bash
# A v2 row's TP8 Commit on an 8x H200 node (vy-glm53-eu, or a preemptible one: vy-glm53-h8) at this tree's default scheme, rows-sha512 under the SHA-512 TP run root
# (`verity/tp-run-root/v2`), with REPLAY_DEFERRED=1 RETAIN=host PAIRS=1: every rank seals a replay bundle of its committed store
# and the Commit ends PENDING until `tp-commit-replay` decides it on a CPU node (replay_publish.sh).  Its row dir holds the Build of
# record and the passing Match (copied in from the CPU node, README "GPU batch"); the 8 GPUs come from the node's lease once every GPU
# reads 0 MiB and no preemptible lease is held.  The bundles and "_rowmeta" (the row dir's small records) are preserved in the
# store, art: ids in $OUT/bundles.json.  The engine's gpu_memory_utilization read back from its vllm_config, the driver's and
# each rank's, goes to $OUT/gpu_memory_utilization.json (exit 12, after the bundles are preserved, unless every one is GPU_UTIL).
# An evicted Commit's run fails, and its rerun is a whole new attempt: the unfinished commit/ (and commit.log) moved aside, a new
# commitment and new draws.  Each attempt is a line of the row dir's commit_attempts.jsonl (in "_rowmeta", and $OUT), and its
# number is in the bundles' and the rowmeta's meta.
#   research run --on vy-glm53-eu --mem-gb 600 --timeout 10h --custody-ttl 24h --source . --cwd source --project verity -- env ROW=... BUILD_RUN=... MATCH_RUN=... NODE=vy-glm53-eu bash integrations/vllm/manifests/glm_v2_tree/jobs/commit.sh
# env: ROW, BUILD_RUN, MATCH_RUN, [NODE=$(hostname)], [SW], [REF] (model.sh's defaults), [MEMORY_MAP], [COMMIT_TIMEOUT].
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}" "${MATCH_RUN:?MATCH_RUN}"
TREE=$PWD
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/model.sh"
TREE_SHA=${RESEARCH_SOURCE_SHA:-$(git -C "$TREE" rev-parse HEAD)}
D=$SW/$ROW
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" || exit 2
test -s "$D/manifest.json" || { echo "no manifest.json in $D: copy the Build of record in first"; exit 2; }
test -s "$D/match/tp2_match.json" || { echo "no match/tp2_match.json in $D: copy the passing Match in first"; exit 2; }
[ ! -s "$D/commit/summary.json" ] || { echo "$D/commit already holds a Commit (summary.json)"; exit 3; }
prev=null
if [ -d "$D/commit" ] || [ -f "$D/commit.log" ]; then
  f=commit.failed-$(date -u +%Y%m%dT%H%M%SZ)
  if [ -d "$D/commit" ]; then mv "$D/commit" "$D/$f"; else mkdir "$D/$f"; fi
  [ ! -f "$D/commit.log" ] || mv "$D/commit.log" "$D/$f/row-commit.log"
  prev="\"$f\"" && echo "an earlier attempt's unfinished commit/ moved aside to $f"
fi
L=$D/commit_attempts.jsonl
ATTEMPT=$(( $(cat "$L" 2>/dev/null | wc -l) + 1 ))
printf '{"attempt": %d, "run": "%s", "node": "%s", "tree": "%s", "started": "%s", "after": %s}\n' "$ATTEMPT" "$RESEARCH_RUN_ID" \
  "${NODE:-$(hostname)}" "$REF@$TREE_SHA" "$(date -u +%FT%TZ)" "$prev" >>"$L" && cp "$L" "$OUT/commit_attempts.jsonl"
echo "Commit attempt $ATTEMPT of $ROW ($L)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$OUT/gpus_at_start.txt"
df -h /workspace | tail -1 | tee "$OUT/disk_at_start.txt"
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
ckpt_bind
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM MATCH_PHASE
export PY=/workspace/cp/venv312/bin/python HF_HOME=/workspace/cp/hf GPU_UTIL=0.88 COMMIT_GPU_UTIL=0.88
export REPLAY_DEFERRED=1 RETAIN=host PAIRS=1 COMMIT_TIMEOUT=${COMMIT_TIMEOUT:-28800}
[ -z "${MEMORY_MAP:-}" ] || export EXTRA_COMMIT_ARGS="--memory-map $MEMORY_MAP"
W=$(mktemp -d /tmp/$V2-commit.XXXXXX); trap 'rm -rf "$W"' EXIT
cat >"$W/gpu-lease" <<'EOF'
#!/usr/bin/env bash
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 != 0 {b = 1} END {exit !b}' \
  || /usr/local/bin/gpu-lease status | grep -q ' held .*preempt=1 '; do sleep 30; done
exec /usr/local/bin/gpu-lease "$@"
EOF
chmod +x "$W/gpu-lease"
export GPU_LEASE=$W/gpu-lease GPU_LEASE_N=8 GPU_LEASE_WHO=circuits-$V2 CUDA_VISIBLE_DEVICES=
export GUARD_PRESERVE_FILE=$SW/guard.preserve TMPDIR=$W
source /workspace/cp/venv312/bin/activate
export PATH=$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$W/research" && chmod +x "$W/research"
export RESEARCH_BIN=$W/research RB=$W/research
compgen -e | grep -E '^(VLLM_|NCCL_|CUDA_|TORCH_|GPU_LEASE|GPU_UTIL|COMMIT_|REPLAY_|RETAIN|PAIRS)' | sort | tr '\n' ' ' | tee "$OUT/engine_env_names.txt"; echo
$PY -m verity_vllm.pipeline.cli row chain --stages commit --build-run "$BUILD_RUN" --match-run "$MATCH_RUN" --no-publish \
  --ref "$REF" --by circuits "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$SW" >"$OUT/chain.out" 2>&1
rc=$?
for f in stages.txt row.log commit.log gpu_lease.json; do cp "$D/$f" "$OUT/" 2>/dev/null; done
for f in summary.json replay_deferred.json runs.jsonl value_check.json coverage.json xrank_collectives.json weights_pin.json fold_match_binding.json; do
  cp "$D/commit/$f" "$OUT/commit_$f" 2>/dev/null
done
grep -E '^\[tp2commit\] ' "$D/commit.log" 2>/dev/null | tail -60 | cut -c1-400 >"$OUT/commit_tagged.txt"
tail -30 "$OUT/chain.out"
python3 "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/gpu_util_read_back.py" "$OUT/gpu_memory_utilization.json" "$GPU_UTIL" \
  "$D/commit/versions.json"; urc=$?
[ "$rc" = 0 ] || exit $rc
ls -d "$D"/commit/replay_bundle_p*_rank* >/dev/null 2>&1 || { echo "no replay bundles in $D/commit: nothing deferred"; exit 5; }
"$RB" keep "$D/commit" --owner @circuits --expires 7d --priority high \
  --why "$WHAT v2 TP8 Commit $RESEARCH_RUN_ID: the ranks' replay bundles for tp-commit-replay and the record" || echo "research keep failed (continuing)"
du -sh "$D"/commit/replay_bundle_p*_rank* | tee "$OUT/bundles_du.txt"
echo '{' >"$OUT/bundles.json"; first=1; prc=0
for b in "$D"/commit/replay_bundle_p*_rank*; do
  [ -f "$b/MANIFEST.json" ] || { echo "$b: not sealed"; prc=5; continue; }
  name=$(basename "$b"); sha=$(sha256sum "$b/MANIFEST.json" | cut -c1-64)
  art=$("$RB" data put --kind vllm-replay-bundle/v1 --tree "$b" --preserve \
        --meta "{\"row\": \"$ROW\", \"bundle\": \"$name\", \"manifest_sha256\": \"$sha\", \"commit_run\": \"$RESEARCH_RUN_ID\", \"attempt\": $ATTEMPT, \"tree\": \"$REF@$TREE_SHA\"}" \
        2>>"$OUT/bundles_put.log" | grep -oE 'art:[0-9a-f]{64}' | tail -1)
  [ -n "$art" ] || { echo "$name: research data put failed (bundles_put.log)"; prc=5; }
  [ $first = 1 ] || echo ',' >>"$OUT/bundles.json"; first=0
  printf ' "%s": {"art": "%s", "manifest_sha256": "%s"}' "$name" "$art" "$sha" >>"$OUT/bundles.json"
  echo "$name $art"
done
M=$(mktemp -d "$SW/.rowmeta.XXXXXX")
(cd "$D" && find . -type f -not -path './build/*' -not -path './aot_trace/*' -not -path './match/capture/*' -not -path './match/control/*' \
   -not -path './match.*/*' -not -path './match/fold_match.failed-*/*' -not -path './commit/replay_bundle_*' -not -path './commit.failed-*/*' -not -name manifest.json \
   -not -regex './match/fold_match/rank[0-9]*/\(instances\.jsonl\|accesses\.jsonl\.gz\|program\.json\)' -print0 | xargs -0 cp --parents -l -t "$M")
echo "rowmeta: $(find "$M" -type f | wc -l) files, $(du -sh --apparent-size "$M" | cut -f1)"
art=$("$RB" data put --kind vllm-row-meta/v1 --tree "$M" --preserve \
      --meta "{\"row\": \"$ROW\", \"commit_run\": \"$RESEARCH_RUN_ID\", \"attempt\": $ATTEMPT, \"tree\": \"$REF@$TREE_SHA\", \"what\": \"the row dir's records less the Build, the capture planes, a moved-aside Match, the folds' instances and the replay bundles\"}" \
      2>>"$OUT/bundles_put.log" | grep -oE 'art:[0-9a-f]{64}' | tail -1)
rm -rf "$M"
[ -n "$art" ] || { echo "rowmeta: research data put failed (bundles_put.log)"; prc=5; }
printf ',\n "_rowmeta": {"art": "%s"}\n}\n' "$art" >>"$OUT/bundles.json"
cat "$OUT/bundles.json"
[ "$prc" != 0 ] || [ "$urc" = 0 ] || { echo "the engine did not read back gpu_memory_utilization=$GPU_UTIL ($OUT/gpu_memory_utilization.json)"; prc=12; }
exit $prc
