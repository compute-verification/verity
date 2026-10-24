#!/usr/bin/env bash
# R1 on the GPUs gpu-lease gave this run: each arm of ARMS ("name:es.py args;name:args;...") on GPU (i / NPG), outputs in
# $OUT/<name>/. es.py imports A3's newskill.py, sent beside it.
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
S=$(mktemp -d /tmp/r1.XXXXXX)
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$IN"/es.py "$IN"/newskill.py "$S"/
cd "$S"
PY=/workspace/jobs/tvi/venv-r1/bin/python
export HF_HUB_CACHE=/workspace/hf/hub HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host=$(hostname) $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
IFS=';' read -ra ARM_LIST <<< "$ARMS"
i=0
pids=()
for spec in "${ARM_LIST[@]}"; do
  name=${spec%%:*}
  args=${spec#*:}
  g=${GPUS[$((i / ${NPG:-2}))]}
  mkdir -p "$OUT/$name"
  echo "arm $name on GPU $g: $args"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g "$PY" es.py $args --out "$OUT/$name" > "$OUT/$name/stdout.txt" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "arms done rc=$rc $(date -u +%FT%TZ)"
for spec in "${ARM_LIST[@]}"; do tail -3 "$OUT/${spec%%:*}/stdout.txt"; done
exit $rc
