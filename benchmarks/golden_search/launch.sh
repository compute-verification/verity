#!/usr/bin/env bash
# One CPU job: N shards of a grid under the scenarios named, P at a time, results under $OUT/res.
#   launch.sh NAME GRID N P "SCENARIOS" [MODEL] [MAX_MIN]
# WHERE=auto (default) is `research run --on auto`; WHERE=pool is node 2's research pool (node 2 takes runs only through
# its check slots), for when `--on auto`'s pick has a longer line.
set -euo pipefail
name=$1 grid=$2 n=$3 p=$4 scen=$5 model=${6:-stub} max=${7:-60}
case ${WHERE:-auto} in
  auto) where=(--on auto) ;;
  pool) where=(--on vy-nebius-2 --pool research --as compute-research --project verity --source "$(git rev-parse --show-toplevel)" --cwd source --timeout $((max * 60))) ;;
  *) echo "WHERE is auto or pool" >&2; exit 2 ;;
esac
cpus=(--cpus "$p" --max-min "$max"); [ "${WHERE:-auto}" = pool ] && cpus=()
exec research run "${where[@]}" "${cpus[@]}" --mem-gb 16 -- bash -c "
  cd benchmarks/golden_search && mkdir -p \$OUT/res && \
  seq 0 $((n - 1)) | xargs -P $p -I{} python3 search.py run --grid $grid --shard {}/$n --scenario $scen \
    --model $model --out \$OUT/res > \$OUT/$name.log 2>&1; tail -5 \$OUT/$name.log"
