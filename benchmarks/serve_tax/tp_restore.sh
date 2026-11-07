#!/usr/bin/env bash
# A model served mid-traffic, restored at its first edge into a fresh engine and replayed to its second (tp_restore.py),
# on one node under `research run`, from the shipped tree's root, the C4 shard sent with --send (or C4=PATH):
#   [MODEL_ID=glm53] [RATE=10] [SPAN_S=240] [EDGES_AT=150,210] [SAMPLING=production] [MAX_MODEL_LEN=N]
#     [SAVER=MODULE:FUNCTION] [LOADER=MODULE:FUNCTION] [PERMUTE=1] [KEEP=1] [MAX_MIN=150]
#     bash benchmarks/serve_tax/tp_restore.sh
# The plan (OUT/plan.jsonl) is made before the GPUs are held; then one server lifetime serves it with the edges into
# OUT/serve, and one in-process engine restores at the first edge and replays to the second into OUT/restore (its verdict
# OUT/restore/restore.json). PERMUTE=1 restores again with the GPUs in reverse order into OUT/restore-permuted. Each edge's
# blocks (tens of GB at TP8) go to a scratch directory outside the run's, removed at the end unless KEEP=1. The GPUs are
# held under gpu-lease where the node has it. VENV's vLLM must be d9105ea80, the integration's pin.
# MODEL_ID=moonlight TP=2 is a smoke of the same path on two ranks.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID=${MODEL_ID:-glm53}
case $MODEL_ID in
  moonlight) CK=${CHECKPOINT:-/workspace/research/scratch/replay-determinism/moonlight}; GPUS=${TP:-1} ;;
  glm53)
    REV=aca966e4e02791568aa6a4ced368624b3d897f42
    if [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then CK=/workspace/cp/glm53-ckpt/$REV; else CK=/workspace/fs/glm53/$REV; fi
    CK=${CHECKPOINT:-$CK}; GPUS=8 ;;
  *) echo "tp-restore: MODEL_ID is moonlight or glm53, not $MODEL_ID"; exit 2 ;;
esac
[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312   # a dc node's, else pod_bootstrap.sh's
VENV=${VENV:-$venv}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
RATE=${RATE:-10}; SPAN_S=${SPAN_S:-240}; EDGES_AT=${EDGES_AT:-150,210}; SAMPLING=${SAMPLING:-production}
DOCS=${DOCS:-45000}; MAX_MIN=${MAX_MIN:-150}; PLAN=$OUT/plan.jsonl
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export HF_HOME=${HF_HOME:-/workspace/cp/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
[ -f "$C4" ] || { echo "tp-restore: no C4 shard (--send it, or C4=PATH)"; exit 2; }
[ -d "$CK" ] || { echo "tp-restore: no checkpoint at $CK"; exit 2; }
v=$("$PY" -c 'import vllm; print(vllm.__version__)') || exit 2
case $v in *+gd9105ea8*) ;; *) echo "tp-restore: vLLM $v is not d9105ea80, the integration's pin"; exit 2 ;; esac
mkdir -p "$OUT"
[ "${1:-}" = leased ] || "$PY" "$here/tp_restore.py" preflight --model-id "$MODEL_ID" --checkpoint "$CK" \
  ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} ${TP:+--tp "$TP"} || exit 2

if [ "${1:-}" = leased ]; then
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.total --format=csv
  echo "== serve $(date -u +%FT%TZ)"
  "$PY" "$here/tp_restore.py" serve --model-id "$MODEL_ID" --checkpoint "$CK" --c4 "$C4" --plan "$PLAN" --docs "$DOCS" \
    --rate "$RATE" --span-s "$SPAN_S" --edges-at "$EDGES_AT" --sampling "$SAMPLING" \
    ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} ${TP:+--tp "$TP"} ${SAVER:+--saver "$SAVER"} \
    --out "$OUT/serve" --scratch "$SCRATCH/serve" || exit 1
  rc=0
  for arm in restore ${PERMUTE:+restore-permuted}; do
    echo "== $arm $(date -u +%FT%TZ)"
    flag=(); env=()
    if [ "$arm" = restore-permuted ]; then
      flag=(--rank-map-free)
      env=(CUDA_VISIBLE_DEVICES="$(echo "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}" | tr , '\n' | tac | paste -sd,)")
    fi
    env "${env[@]}" "$PY" "$here/tp_restore.py" restore --served "$OUT/serve" --scratch "$SCRATCH/$arm" --out "$OUT/$arm" \
      ${LOADER:+--loader "$LOADER"} "${flag[@]}" > "$OUT/$arm.log" 2>&1 || { tail -n 40 "$OUT/$arm.log"; rc=1; }
    tail -n 1 "$OUT/$arm.log"
  done
  exit $rc
fi

"$PY" "$here/serve_tax.py" plan --model-id "$MODEL_ID" --checkpoint "$CK" --c4 "$C4" --rates "load=$RATE" \
  --span-s "$SPAN_S" --docs "$DOCS" --out "$PLAN" || exit 1
root=/workspace/research/scratch; [ -d "$root" ] || root=${TMPDIR:-/tmp}
SCRATCH=${SCRATCH:-$(mktemp -d "$root/tp-restore.XXXXXX")}
export SCRATCH
if command -v gpu-lease > /dev/null; then
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-tp-restore} gpu-lease "$GPUS" --wait --max-min "$MAX_MIN" -- bash "$here/tp_restore.sh" leased
else
  bash "$here/tp_restore.sh" leased   # a node without gpu-lease: its jobs queue by other means (one at a time)
fi
rc=$?
gzip -f "$OUT"/serve/server/step_trace.*.jsonl 2>/dev/null
du -sh "$SCRATCH" 2>/dev/null
[ "${KEEP:-0}" = 1 ] || rm -rf "$SCRATCH"
exit $rc
