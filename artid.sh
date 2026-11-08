#!/usr/bin/env bash
# artid.sh RUN...: each run's published outputs (its attempt record in this node's store).
for r in "$@"; do
  f=/workspace/research/store/attempts/$r.json
  echo "== $r"
  [ -f "$f" ] && python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('outputs'),indent=1))" "$f"
done
