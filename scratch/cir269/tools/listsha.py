"""The coordinator's listing summary (r20261113-072414-ff5e's inputs/bindings-list.sh) of each bindings.json, and how two
differ by target: python3 listsha.py A.json [B.json]"""
import hashlib
import json
import sys


def targets(path):
    d = json.load(open(path))
    b = d["bindings"]
    t = sorted({(k, l) for v in b.values() for k, l in v})
    print(path)
    print("  bindings", len(b), "targets", len(t), "root_errors", len(d["root_errors"]), "unchecked", len(d["unchecked"]))
    print("  targets sha256", hashlib.sha256("\n".join(f"{k} {l}" for k, l in t).encode()).hexdigest())
    return set(t), set(b)


ts = [targets(p) for p in sys.argv[1:]]
if len(ts) == 2:
    (a, ab), (b, bb) = ts
    print("targets only in first", len(a - b), sorted(a - b)[:10])
    print("targets only in second", len(b - a), sorted(b - a)[:10])
    print("bindings only in first", sorted(ab - bb)[:10], "only in second", sorted(bb - ab)[:10])
