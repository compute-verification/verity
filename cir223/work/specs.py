#!/usr/bin/env python3
"""specs.py SPEC...: N (a head merged with the commit below the move, else the raw head), N@P (child N merged with parent P's base merge),
N@P:onto, Nraw (the raw head), Nown (a child's own base merge)."""
import json, sys
b = json.load(open('/tmp/cir223b/work/base-merge.json')); c = json.load(open('/tmp/cir223b/work/child-merge.json'))
out = []
for s in sys.argv[1:]:
    if '=' in s:
        out.append(s)
        continue
    n, _, rest = s.partition('@')
    if n.endswith('raw'):
        out.append(f"{n}={b[n[:-3]]['head']}")
    elif n.endswith('own'):
        out.append(f"{n}={b[n[:-3]]['merged']}")
    elif rest:
        p, _, how = rest.partition(':')
        assert c[n]['parent'] == p
        out.append(f"{n}{how}={c[n]['merged']}@{p}" + (f":{how}" if how else ""))
    else:
        out.append(f"{n}={b[n].get('merged') or b[n]['head']}")
print(' '.join(out))
