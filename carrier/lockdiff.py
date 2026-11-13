"""Which records of a lock differ between two copies, and in which fields.

    python3 lockdiff.py COMMITTED.json REGENERATED.json

Section by section: keys only in one, and for a key in both, the fields whose values differ. For a guarantee's record it says
whether `signature`, `type_hash` or `unfolded_hash` (the guarantee's own record) is among them.
"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
OWN = {"signature", "type_hash", "unfolded_hash"}
total = 0
for sec in sorted(set(a) | set(b)):
    x, y = a.get(sec), b.get(sec)
    if x == y:
        continue
    if isinstance(x, dict) and isinstance(y, dict):
        only_a, only_b = sorted(set(x) - set(y)), sorted(set(y) - set(x))
        diff = sorted(k for k in set(x) & set(y) if x[k] != y[k])
        print(f"[{sec}] only in committed: {len(only_a)} {only_a[:6]}; only in regenerated: {len(only_b)} {only_b[:6]}; differ: {len(diff)}")
        for k in diff:
            total += 1
            if isinstance(x[k], dict) and isinstance(y[k], dict):
                fields = sorted(f for f in set(x[k]) | set(y[k]) if x[k].get(f) != y[k].get(f))
                own = sorted(set(fields) & OWN)
                print(f"   {k}: fields {fields}{'  OWN RECORD: ' + str(own) if own else ''}")
                for f in fields:
                    u, v = x[k].get(f), y[k].get(f)
                    if isinstance(u, dict) and isinstance(v, dict):
                        sub = sorted(g for g in set(u) | set(v) if u.get(g) != v.get(g))
                        print(f"      {f}: {len(sub)} entries differ: {sub[:8]}")
                    elif isinstance(u, list) and isinstance(v, list):
                        print(f"      {f}: -{[i for i in u if i not in v][:6]} +{[i for i in v if i not in u][:6]}")
                    else:
                        print(f"      {f}: {str(u)[:70]} -> {str(v)[:70]}")
            else:
                print(f"   {k}: {str(x[k])[:80]} -> {str(y[k])[:80]}")
    else:
        print(f"[{sec}] differs (not a table): {str(x)[:100]} -> {str(y)[:100]}")
print(f"records that differ: {total}")
