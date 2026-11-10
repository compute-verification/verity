#!/bin/bash
# status.sh RUN...: each run's state, and its result when it has ended (fetched into ~/.research/runs/RUN).
cd "$(dirname "$0")" || exit 1
for r in "$@"; do
  /workspace/.venv/bin/research status "$r" --json 2>/dev/null | python3 -c '
import json, sys
x = json.load(sys.stdin)["runs"][0]
c = x.get("command") or []
i = c.index("--") if "--" in c else 0
print(x["id"], x.get("class"), "rc=%s" % x.get("rc"), (x.get("remote") or {}).get("machine"), "ended=%s" % x.get("ended_utc"),
      "wall_min=%.2f" % ((x.get("cost") or {}).get("wall_s", 0) / 60), "|", " ".join(c[i + 1:])[-150:])'
  if /workspace/.venv/bin/research status "$r" --json 2>/dev/null | grep -q '"ended_utc": "2'; then
    [ -d ~/.research/runs/"$r"/out ] || /workspace/.venv/bin/research fetch "$r" --all >/dev/null 2>&1
    tail -n "${N:-2}" ~/.research/runs/"$r"/stdout.log | cut -c1-400
  fi
done
