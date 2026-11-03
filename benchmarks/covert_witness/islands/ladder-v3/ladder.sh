#!/usr/bin/env bash
# ladder.sh SIZE BATCH WORK [FLAGS...]: one size's v3 ladder run (ladder.py), inside `gpu-lease N`: one step counted on the
# first leased GPU (ladder.py count, kept in WORK), then ladder.py train on every leased GPU with its outputs in $OUT.
# WORK (persistent) holds the checkpoint and the registered starts, so a rerun with the same flags resumes, on any
# number of GPUs that divides BATCH by --micro (--stop-at or --yield-free ends a phase checkpointed).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pyenv.sh"
DATA=${LADDER_DATA:-/workspace/research/scratch/covert-witness/train/v3}
size=$1 batch=$2 work=$3
shift 3
IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
n=${#gpus[@]}
[ "$n" -ge 1 ] || { echo "ladder.sh: no GPU leased (CUDA_VISIBLE_DEVICES is empty)" >&2; exit 2; }
mkdir -p "$OUT" "$work"
PYTHONPATH="$here/../../../../tools/research/src" python3 -m research keep "$work" --owner @training --expires never \
  --priority keep --why "the v3 starts ladder's $size run: its checkpoint (a resume) and registered starts (islands/ladder-v3/ladder.sh)" \
  || echo "ladder.sh: WARNING: no retention record for $work" >&2
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
if [ ! -s "$work/count.json" ]; then
  # LADDER_COUNT_MICRO: a smaller micro-batch for the eager count, whose activations a compiled run doesn't keep
  CUDA_VISIBLE_DEVICES=${gpus[0]} "$P" "$here/ladder.py" count --size "$size" --batch "$batch" --out "$work" "$@" \
    ${LADDER_COUNT_MICRO:+--micro "$LADDER_COUNT_MICRO"}
fi
cp "$work/count.json" "$OUT/count.json"
exec "$P" -m torch.distributed.run --standalone --nproc_per_node "$n" "$here/ladder.py" train --size "$size" \
  --batch "$batch" --data "$DATA" --work "$work" --out "$OUT" --count "$work/count.json" "$@"
