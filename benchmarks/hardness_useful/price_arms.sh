#!/usr/bin/env bash
# Price QAT arms' codes as each arm finishes, on the node that holds them (`research run --pool research`, cwd source):
#   bash benchmarks/hardness_useful/price_arms.sh WORKERS RUN:NAME...
# An arm is done when its run's result.json has its final "seconds"; an arm whose run ended without it is skipped.
# Each pricing goes to $OUT/NAME (price.py's defaults: depth 4, every correction price, the K and n orders).
set -uo pipefail
workers=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
runs=/workspace/research/runs
mkdir -p "${OUT:?}"
pending=("$@")
rc=0
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for arm in "${pending[@]}"; do
    run=${arm%%:*}; name=${arm#*:}
    res=$runs/$run/out/$name/result.json
    state=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["transitions"][-1]["state"])' \
      "$runs/$run/status.json" 2>/dev/null || echo unknown)
    if [ -f "$res" ] && python -c 'import json,sys; sys.exit(0 if "seconds" in json.load(open(sys.argv[1])) else 1)' "$res"; then
      echo "$(date -u +%H:%M:%S) pricing $name ($run)"
      python -u "$here/price.py" --codes "$HU_BASE/codes/$run/$name.safetensors" --full-check --workers "$workers" \
        --out "$OUT/$name" > "$OUT/$name.log" 2>&1 || { echo "price $name failed"; rc=1; }
      tail -1 "$OUT/$name.log"
    elif [ "$state" != running ] && [ "$state" != waiting ] && [ "$state" != unknown ]; then
      echo "$(date -u +%H:%M:%S) $name ($run) ended $state without a result: skipped"; rc=1
    else
      left+=("$arm")
    fi
  done
  pending=("${left[@]+"${left[@]}"}")
  [ ${#pending[@]} -gt 0 ] && sleep 60
done
exit $rc
