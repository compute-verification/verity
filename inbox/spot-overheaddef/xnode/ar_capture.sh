#!/bin/bash
# The GLM-5.3 TP8 all-reduce capture on this 8-GPU node, under research run --cwd source (a tree at the capture's commit):
#   HOLD=/workspace/research/holds/glm53-sessions [IDLE_MIN=5] [CAP_MIN=26] [ONLY=...] bash "$RESEARCH_RUN_DIR/inputs/ar_capture.sh"
# HOLD is written first (vllm's session job waits while it exists; a HOLD already there refuses the run, exit 6), then every
# GPU must be under 1 GiB used within IDLE_MIN; the capture runs under `timeout` in its own process group (its rank and
# compare processes die with it), and HOLD is removed on every exit, when it still holds this run's line.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; mkdir -p "$OUT"
SRC=$(pwd -P)
[ -f "$SRC/integrations/vllm/tests/properties/glm53_tp8_allreduce_capture_gpu.py" ] || { echo "ar-capture: run with --cwd source"; exit 2; }
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
ONLY=${ONLY:-fused,allreduce,probe,again,nccl_probe,nccl_again}
TAG="kernels all-reduce capture ($(basename "$D")): sessions wait"
SCR=$(mktemp -d /workspace/ar-capture-scratch.XXXXXX) || exit 2
release() {
  rm -rf "$SCR"
  if [ -n "${HOLD:-}" ] && [ -f "$HOLD" ] && grep -qF "$TAG" "$HOLD"; then rm -f "$HOLD" && echo "ar-capture: removed $HOLD $(date -u +%FT%TZ)"; fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$OUT/gpus_after.csv"
}
trap release EXIT
trap 'exit 143' TERM INT HUP
echo "ar-capture: tree $(git -C "$SRC" rev-parse HEAD 2>/dev/null) start $(date -u +%FT%TZ)"
if [ -n "${HOLD:-}" ]; then
  [ -f "$HOLD" ] && { echo "ar-capture: $HOLD is held: $(head -c 300 "$HOLD")"; HOLD=; exit 6; }
  mkdir -p "$(dirname "$HOLD")" && echo "$TAG" > "$HOLD" || exit 2
  echo "ar-capture: wrote $HOLD $(date -u +%FT%TZ)"
fi
for i in $(seq 0 $((${IDLE_MIN:-5} * 6))); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)
  [ "$busy" = 0 ] && break
  [ "$i" = $((${IDLE_MIN:-5} * 6)) ] && { echo "ar-capture: $busy GPUs still in use after ${IDLE_MIN:-5} min"; nvidia-smi; exit 5; }
  sleep 10
done
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | tee "$OUT/gpus_before.csv"
df -BG /workspace | tail -1
cd "$SRC/integrations/vllm" || exit 2
export PYTHONPATH=$SRC/integrations/vllm:$SRC:$SRC/tools/research/src${PYTHONPATH:+:$PYTHONPATH}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=$VENV/bin:/usr/local/cuda/bin:$PATH
echo "ar-capture: capture start $(date -u +%FT%TZ) only=$ONLY"
timeout --kill-after=60 "${CAP_MIN:-26}m" "$PY" tests/properties/glm53_tp8_allreduce_capture_gpu.py --out "$OUT/capture" \
  --scratch "$SCR" --only "$ONLY" 2>&1 | tee "$OUT/capture.log"
rc=${PIPESTATUS[0]}
echo "ar-capture: capture rc=$rc end $(date -u +%FT%TZ)"
exit "$rc"
