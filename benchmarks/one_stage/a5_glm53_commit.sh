#!/usr/bin/env bash
# One-stage audit, run A5, its GPU half: GLM-5.3's TP8 Commit on an 8xH200, the Commit of record's row (its Build, Match and
# manifest of record, read in place from sweep-b3) Committed once more with the verifier's SiluMul_v2 partition
# (glm53_silu_mul_v2_tp8.json) committed as served windows, one per rank (SERVING_ROWS, REPLAY_DEFERRED=1), one pair, into its
# own sweep dir (sweep-a5): the Commit of record's commit/ is never written.
#   bash benchmarks/one_stage/a5_glm53_commit.sh
#   (a job run from the source tree, up to 12 h, keeping out/**)
# Needs on the node: the row of record at $SRC/$ROW (manifest.json, build/, match/capture/tokens.json), the checkpoint's verified
# local copy at /workspace/cp/glm53-ckpt/$REV (bound over its pinned path /workspace/fs/glm53/$REV, created if absent),
# /workspace/cp/venv312 and /workspace/cp/hf. The windows' private bodies (m*-private.bin, v*-private.bin: rows and salts)
# leave the run dir for HANDOFF before the job ends, so custody never uploads them; $OUT keeps the public files, the TP
# Commit's record and handoff.txt (HANDOFF's path), which a5_glm53_cpu.sh copies to the CPU machine.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
SRC=/workspace/fs/glm53-failfast/sweep-b3
BASE=${GLM_BASE:-/workspace/fs/glm53-failfast/sweep-a5}
ROW=glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
REV=aca966e4e02791568aa6a4ced368624b3d897f42
TREE=$PWD
HANDOFF=/workspace/research/glm53-a5/handoff-${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
test -s "$SRC/$ROW/manifest.json" || { echo "no manifest.json in $SRC/$ROW"; exit 2; }
test -s "$SRC/$ROW/match/capture/tokens.json" || { echo "no match/capture/tokens.json in $SRC/$ROW"; exit 2; }
# the A5 row dir: the Build and the Match of record by link (read only), a copy of the manifest of record (the commit stage rewrites
# a manifest it does not hold of record, and a copy keeps that write off the original)
mkdir -p "$BASE/$ROW"
for x in build match; do [ -e "$BASE/$ROW/$x" ] || ln -s "$SRC/$ROW/$x" "$BASE/$ROW/$x"; done
cmp -s "$SRC/$ROW/manifest.json" "$BASE/$ROW/manifest.json" 2>/dev/null || cp "$SRC/$ROW/manifest.json" "$BASE/$ROW/manifest.json"
cmp -s "$SRC/$ROW/manifest.json" "$BASE/$ROW/manifest.json" || { echo "the manifest copy differs from the manifest of record"; exit 2; }
cp "$SRC/$ROW/build_summary.json" "$BASE/$ROW/" 2>/dev/null
ls -la "$BASE/$ROW" | tee "$OUT/row-dir.txt"
nvidia-smi -L | tee "$OUT/gpus.txt"
FS_CK=/workspace/fs/glm53/$REV; LOCAL_CK=/workspace/cp/glm53-ckpt/$REV
if [ -e "$LOCAL_CK/.verified" ] && ! mountpoint -q "$FS_CK"; then
  mkdir -p "$FS_CK" && sudo -n mount --bind "$LOCAL_CK" "$FS_CK" && echo "checkpoint: local verified copy bound over $FS_CK"
fi
mountpoint -q "$FS_CK" && echo "checkpoint: $FS_CK is a mount (local copy)" || echo "checkpoint: $FS_CK read over the shared fs"
test -s "$FS_CK/config.json" || { echo "no checkpoint at $FS_CK (and no verified local copy at $LOCAL_CK)"; exit 2; }
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export GPU_UTIL=0.92 COMMIT_GPU_UTIL=0.92 HF_HOME=/workspace/cp/hf PY=/workspace/cp/venv312/bin/python
export SERVING_ROWS=$TREE/benchmarks/one_stage/glm53_silu_mul_v2_tp8.json REPLAY_DEFERRED=1 PAIRS=${PAIRS:-1} COMMIT_TIMEOUT=${COMMIT_TIMEOUT:-10800}
export TMPDIR=$(mktemp -d /tmp/glm53-commit-a5.XXXXXX)
source /workspace/cp/venv312/bin/activate
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/tools/config:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/benchmarks/numerical/python:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
cp "$SERVING_ROWS" "$OUT/partition.json"
uv run --frozen --no-sync --active --project "$TREE" verity-vllm row run "$ROW" GLM53 zai-org/GLM-5.3 "$REV" \
  --stages commit --sweep-dir "$BASE" > "$OUT/commit.out" 2>&1
rc=$?
for f in stages.txt row.log commit.log manifest.log; do cp "$BASE/$ROW/$f" "$OUT/" 2>/dev/null; done
cp "$BASE/$ROW/commit/summary.json" "$BASE/$ROW/commit/verdict.json" "$OUT/" 2>/dev/null
# the windows: private bodies to HANDOFF (off custody), public files and the TP Commit's record to $OUT
RD=${RESEARCH_RUN_DIR:-.}/serving-rows/$ROW
mkdir -p "$HANDOFF/commit" "$OUT/serving-rows" "$OUT/commit-record"
cp "$BASE/$ROW/commit/runs.jsonl" "$BASE/$ROW/commit/versions.json" "$HANDOFF/commit/" 2>/dev/null
cp "$HANDOFF/commit/"* "$OUT/commit-record/" 2>/dev/null
if [ -d "$RD" ]; then
  mv "$RD" "$HANDOFF/serving-rows"
  for d in "$HANDOFF/serving-rows"/rank*; do
    mkdir -p "$OUT/serving-rows/$(basename "$d")"
    cp "$d"/*.json "$d"/m*-public.bin "$d"/error.txt "$OUT/serving-rows/$(basename "$d")/" 2>/dev/null
  done
  cp "$HANDOFF/serving-rows"/*.json "$OUT/serving-rows/" 2>/dev/null
fi
rkeep() { $PY -c 'import sys; from research.cli import main; sys.exit(main())' keep "$@" 2>&1 | tail -2; }
rkeep "$HANDOFF" --owner @circuits --expires 7d --priority high \
  --why "A5: GLM-5.3 TP8 served windows (private bodies: prover only) and the TP Commit's record, for the CPU audit"
# the A5 Commit's own commit/ (its deferred replay bundles); never the row dir, whose build/ and match/ link to the Commit of record's
rkeep "$BASE/$ROW/commit" --owner @circuits --expires 3d --priority normal \
  --regenerate "research run --on vy-glm53-eu2 ... -- bash benchmarks/one_stage/a5_glm53_commit.sh" \
  --why "A5's GLM-5.3 TP8 Commit with served windows: its record and deferred replay bundles"
du -sh "$HANDOFF" 2>/dev/null; echo "HANDOFF=$HANDOFF" | tee "$OUT/handoff.txt"
cat "$HANDOFF/serving-rows/tp.json" 2>/dev/null | head -c 2000; echo
tail -40 "$OUT/commit.out"
exit $rc
