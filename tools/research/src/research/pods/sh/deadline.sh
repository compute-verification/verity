#!/usr/bin/env bash
# terminate every owned pod at the deadline, self last
PREFIX=$1; DL=$2; ME=$3
while [ $(date +%s) -lt $DL ]; do sleep 30; done
echo "$(date -u +%FT%TZ) DEADLINE reached" >> /root/dm/deadline.log
python3 - "$PREFIX" "$ME" <<'PY' >> /root/dm/deadline.log 2>&1
import sys; sys.path.insert(0, '/root/dm'); import runpod
prefix, me = sys.argv[1], sys.argv[2]
pods = runpod._request('GET', '/pods') or []
mine = [p for p in pods if str(p.get('name','')).startswith(prefix) and (p.get('desiredStatus') or '').upper() not in ('EXITED','TERMINATED')]
for p in sorted(mine, key=lambda p: p['id'] == me):
    try:
        runpod._request('DELETE', f"/pods/{p['id']}"); print('TERMINATED', p['id'], p.get('name'))
    except SystemExit as e:
        print('terminate failed', p['id'], e)
PY
