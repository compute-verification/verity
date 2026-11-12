"""Callee rows of every by-calls root that differ between two reports.  python3 - A B < cir255_rowdiff.py"""
import json, sys
def rows(p):
    d = json.load(open(p))
    return {(t["definition"], r["callee"]): {k: r.get(k) for k in ("binding", "sizes", "verdict", "why")}
            for t in d["targets"] if t.get("mode") == "by-calls" for r in t.get("by_calls", {}).get("callees", [])}
a, b = rows(sys.argv[1]), rows(sys.argv[2])
print(len(a), "rows in", sys.argv[1], ";", len(b), "in", sys.argv[2])
for k in sorted(a.keys() | b.keys()):
    if a.get(k) != b.get(k):
        print(k[0][:40], "|", k[1], "|", a.get(k), "->", b.get(k))
