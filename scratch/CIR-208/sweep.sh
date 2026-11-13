#!/bin/bash
# sweep.sh JOBS [OUT]: the harness on each "NAME SHA" line, two at a time (OUT/, default out/), then each head whose merge
# stopped again with the merged head's side of every conflict (OUT-forced/); one line of JSON a head in sweep.log
cd /tmp/cir208
out=${2:-out}
export out
xargs -P 2 -L 1 sh -c 'python3 check.py "$0" "$1" --out=$out >> /tmp/cir208/sweep.log 2>> /tmp/cir208/sweep.err' < "$1"
echo "PASS1 $out $(date -u +%FT%TZ)" >> /tmp/cir208/sweep.log
python3 - "$1" "$out" > "/tmp/cir208/jobs-$out-forced.txt" <<'PY'
import json, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    name, sha = line.split()[:2]
    p = Path("/tmp/cir208") / sys.argv[2] / f"{name}.json"
    if not p.exists() or json.loads(p.read_text()).get("conflicts"):
        print(name, sha)
PY
xargs -P 2 -L 1 sh -c 'python3 check.py "$0" "$1" --resolve=theirs --force --out=$out-forced >> /tmp/cir208/sweep.log 2>> /tmp/cir208/sweep.err' < "/tmp/cir208/jobs-$out-forced.txt"
echo "DONE $out $(date -u +%FT%TZ)" >> /tmp/cir208/sweep.log
