#!/usr/bin/env bash
# tick.sh: the runs still going, the memory of each shard and of the vLLM run, and whether #9364 and the moves are on main
cd /agent/repos/verity && git fetch -q origin main 2>/dev/null
echo "main $(git rev-parse --short=12 origin/main); #9364 on main: $(git merge-base --is-ancestor 5c3e7f2abcca813afbd7a92cb952aef590080bfe origin/main && echo yes || echo no); served map on main: $(git cat-file -e origin/main:adapters/vllm/verity_vllm/definitions/registry/catalog.py 2>/dev/null && echo yes || echo no)"
bash /tmp/cir269/st.sh 2>&1 | sort -u | grep -v 'failed 143' | grep -v 'done 0'
ids=""
for n in shard1b shard2b shard3b shard4b m1-vllm; do
  f=/tmp/cir269/runs/$n.log
  [ -f "$f" ] || continue
  r=$(grep -oE 'r2026[0-9]{4}-[0-9]{6}-[0-9a-f]{4}' "$f" | head -1)
  [ -n "$r" ] && ids="$ids $r" && echo "$n=$r"
done
bash /tmp/cir269/watch.sh $ids
date -u +%H:%MZ
