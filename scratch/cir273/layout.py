import collections, json, sys
from verity.definitions.registry import sampled_accept as A
from verity.circuits import bind
pp = A.pass_plan(5, 154880)
c = collections.Counter((u.key[0], u.defn.id) for u in pp)
print("pass_plan units", len(pp), "distinct", len({u.defn.id for u in pp}))
for (k, i), n in sorted(c.items()): print(f"  {k:7s} {n:6d}  {i}")
t = bind(A.SampledAccept, K=5, V=154880, TILE=8192)
print("id", t.id)
print("ports", json.dumps([[str(n), str(ty)] for n, ty in t.signature()[0]]), str(t.signature()[1]))
