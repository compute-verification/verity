#!/usr/bin/env bash
# watch.sh RUN...: each run's state, its last staging and verdict lines, and the stager's resident memory
cd /agent/repos/verity
for r in "$@"; do
  uv run -q --project tools/research research fetch $r --all 2>&1 | tail -1 | cut -c11-90
  L=~/.research/runs/$r/out/job.log
  grep -E "^..:..:..Z [a-z0-9-]+: (stage.py|staging refused|--zk session FAILED|others FAILED|Lean not run|[a-z].* N=)|done, rc|lean B" $L 2>/dev/null | cut -c1-${W:-250} | tail -${N:-4}
  python3 -c "
import json;s=json.load(open('/home/ubuntu/.research/runs/$r/slot.json'));print(' slot', s.get('state'), s.get('took') or s.get('why','')[:120])" 2>/dev/null
  for f in ~/.research/runs/$r/out/*/B/stage.rss; do [ -f "$f" ] || continue; echo " $(basename $(dirname $(dirname $f))): $(awk '$2>m{m=$2;t=$1} END{printf "max %.1f GB at %s", m/1048576, t}' $f); last $(tail -1 $f | awk '{printf "%s %.1f GB, %.1f GB staged", $1, $2/1048576, $3/1048576}')"; done
done
date -u +%H:%MZ
