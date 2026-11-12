#!/bin/bash
# For each pytest shard run: fetch --all, the circuit_check suite's row (seconds, workers), its tests' sum, the session's build
# (cir255_wait.py on resources.jsonl, from the store when evicted), then delete the copy.
R="uv run -q --project /agent/repos/verity/tools/research research"
for r in "$@"; do
  d=~/.research/runs/$r
  (cd /agent/repos/verity && timeout 600 $R fetch $r --all >/dev/null 2>&1)
  mkdir -p /tmp/res/$r /tmp/ext/$r
  cp $d/slot.json $d/job.json /tmp/ext/$r/ 2>/dev/null
  cp $d/suites/tools_verity_circuit_check.json /tmp/ext/$r/cc_suite.json 2>/dev/null
  python3 - "$d" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = json.loads((d / "suites.json").read_text())
rows = rows if isinstance(rows, list) else rows.get("suites", [])
row = next((r for r in rows if r.get("suite") == "tools/verity/circuit_check"), {})
p = json.loads((d / "suites/tools_verity_circuit_check.json").read_text()).get("passed") or {}
print(d.name, {k: row.get(k) for k in ("status", "seconds", "workers")}, "tests", len(p), "sum_s", round(sum(p.values()), 1))
EOF
  if [ -f $d/resources.jsonl ]; then cp $d/resources.jsonl /tmp/res/$r/; else
    a=$(python3 -c "import json;p=json.load(open('$d/run-files.evicted.json'));print([f['artifact'] for f in p['files'] if f['path']=='resources.jsonl'][0])")
    (cd /agent/repos/verity && timeout 300 $R data fetch $a --path resources.jsonl --to /tmp/res/$r >/dev/null 2>&1)
    f=$(find /tmp/res/$r -name resources.jsonl | head -1); [ "$f" != /tmp/res/$r/resources.jsonl ] && mv "$f" /tmp/res/$r/resources.jsonl
  fi
  rm -rf $d
  python3 /tmp/cir255_wait.py /tmp/res/$r 2>/dev/null
  python3 /tmp/cir255_apart.py /tmp/res/$r | sed -n '2,5p'
done
