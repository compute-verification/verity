#!/usr/bin/env bash
# One `research run --on auto` job: N shards of a grid under the scenarios named, P at a time, results under $OUT/res.
#   launch.sh NAME GRID N P "SCENARIOS" [MODEL] [MAX_MIN]
set -euo pipefail
name=$1 grid=$2 n=$3 p=$4 scen=$5 model=${6:-stub} max=${7:-60}
exec research run --on auto --cpus "$p" --mem-gb 16 --max-min "$max" -- bash -c "
  cd benchmarks/golden_search && mkdir -p \$OUT/res && \
  seq 0 $((n - 1)) | xargs -P $p -I{} python3 search.py run --grid $grid --shard {}/$n --scenario $scen \
    --model $model --out \$OUT/res > \$OUT/$name.log 2>&1; tail -5 \$OUT/$name.log"
