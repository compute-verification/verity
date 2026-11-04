#!/usr/bin/env bash
# Pricing jobs on a CPU node, in parallel: each line of JOBS is NAME and `hardness_public/rotated.py` arguments, its
# outputs in $OUT/NAME and its log in $OUT/NAME.log; blank and # lines are skipped. Fails if any job fails.
#   bash benchmarks/hardness_defence/price.sh JOBS
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../hardness_public/env.sh"
mkdir -p "${OUT:?}"
pids=()
while read -r name args; do
  if [ -z "$name" ] || [[ $name == \#* ]]; then continue; fi
  eval "python -u '$here/../hardness_public/rotated.py' $args --out '$OUT/$name'" > "$OUT/$name.log" 2>&1 &
  pids+=($!)
done < "$1"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h -A3 '^target' "$OUT"/*.log | head -50 || true
exit $rc
