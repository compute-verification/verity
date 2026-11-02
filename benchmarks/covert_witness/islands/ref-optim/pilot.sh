#!/usr/bin/env bash
# pilot.sh SIZE BATCH TAG CONFIGS: hyperparameter pilots for the ladder, inside `gpu-lease N`: each line of CONFIGS
# (`NAME FLAGS...`, a file in this directory, at most N lines) is one single-GPU ladder.py run scored on the stream's
# val.bin (`--eval pilot`, nothing registered), all at once, each in $OUT/NAME with its log beside it.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
P=${LADDER_PYTHON:-/workspace/jobs/venv312/bin/python}
DATA=${LADDER_DATA:-/workspace/research/scratch/te-ladder/stream}
size=$1 batch=$2 tag=$3 configs=$here/$4
IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
n=${#gpus[@]}
[ "$n" -ge 1 ] || { echo "pilot.sh: no GPU leased" >&2; exit 2; }
mkdir -p "$OUT"
lines=$(grep -cv '^\s*\(#\|$\)' "$configs")
[ "$lines" -le "$n" ] || { echo "pilot.sh: $lines configs, $n GPUs leased" >&2; exit 2; }
i=0
while read -r name flags; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  g=${gpus[$i]}
  work=/workspace/research/scratch/te-ladder/pilots/$tag/$name
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g "$P" "$here/ladder.py" train --size "$size" --batch "$batch" --data "$DATA" --work "$work" \
    --out "$OUT/$name" --eval pilot $flags > "$OUT/$name.log" 2>&1 &
  i=$((i + 1))
done < "$configs"
wait
for f in "$OUT"/*.log; do
  echo "== $(basename "$f" .log)"
  grep '"phase": "annealed"' "$f" | python3 -c "
import json, sys
for l in sys.stdin:
    e = json.loads(l)['eval']; print(e['branch'], round(e['nats_per_token'], 4))"
  tail -n 2 "$f" | grep -v '^{' || true
done
