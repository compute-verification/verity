#!/usr/bin/env bash
# st.sh: each run's state, rc, host and source sha
cd /agent/repos/verity || exit 1
runs="r20261113-050628-0901 r20261113-050628-3ace"
for f in /tmp/cir269/runs/*.log; do
  r=$(grep -oE 'r2026[0-9]{4}-[0-9]{6}-[0-9a-f]{4}' "$f" | head -1)
  [ -n "$r" ] && runs="$runs $r"
done
for r in $runs; do
  timeout 60 research status "$r" --json 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)['runs'][0]
except Exception as e:
    print('$r', 'unreadable', e); sys.exit()
rem=d.get('remote') or {}
print('$r', d.get('state'), d.get('rc'), (rem.get('host') or '')[:14], (rem.get('source_sha') or '')[:12])"
done
