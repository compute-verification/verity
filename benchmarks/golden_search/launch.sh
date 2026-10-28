#!/usr/bin/env bash
# One CPU job: N shards of a grid under the scenarios named, P at a time, results under $OUT/res.
#   launch.sh NAME GRID N P "SCENARIOS" [MODEL] [MAX_MIN]
# WHERE=auto (default) is `research run --on auto`; WHERE=queue queues it on node 2's share (the queue `--on auto` falls back
# to), for when every pool has a line and node 2's cores are free.
set -euo pipefail
name=$1 grid=$2 n=$3 p=$4 scen=$5 model=${6:-stub} max=${7:-60}
case ${WHERE:-auto} in
  auto) where=(--on auto) ;;
  queue) where=(--queue --source "$(git rev-parse --show-toplevel)" --project verity --on vy-nebius-2 --cwd source) ;;
  *) echo "WHERE is auto or queue" >&2; exit 2 ;;
esac
exec research run "${where[@]}" --cpus "$p" --mem-gb 16 --max-min "$max" -- bash -c "
  cd benchmarks/golden_search && mkdir -p \$OUT/res && \
  seq 0 $((n - 1)) | xargs -P $p -I{} python3 search.py run --grid $grid --shard {}/$n --scenario $scen \
    --model $model --out \$OUT/res > \$OUT/$name.log 2>&1; tail -5 \$OUT/$name.log"
