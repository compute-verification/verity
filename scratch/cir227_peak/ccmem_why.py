"""A measure for one run, on a scratch branch only: for each catalog binding, whether its record in CACHE counts as
circuit-check's test session looks it up (`units._lookup` under conftest's two patches), and for one that misses, which
condition fails.

    ccmem_why.py CACHE        WHY_ROOT=0: without the tracer-root patch
"""
import collections
import json
import os
import sys
import time
from pathlib import Path

from circuit_check import cache as CC, cli, targets as TG, units as UN

cache = Path(sys.argv[1])
CC.UNKEYED = CC.UNKEYED | {k for k in os.environ if k.startswith("VERITY_SUITE_")} | {"CIRCUIT_CHECK_RECORDS"}
if os.environ.get("WHY_ROOT", "1") != "0":
    CC._ROOT = [cli.REPO.resolve()]
TG.load_registries()
t = time.time()
entries = UN.made_entries()
why, kinds, names, ex, by_kind = collections.Counter(), collections.Counter(), collections.Counter(), {}, collections.Counter()
counted = 0


def differs(kind, a, b, had_a, had_b):
    if not (had_a and had_b):
        return True
    if kind in ("modules", "units"):
        return UN._sited(a) != UN._sited(b)
    if kind == "present":
        return (a is None) != (b is None)
    if kind == "enums":
        return (a is None) != (b is None) or (a is not None and a.keys() != b.keys())
    return a != b


for e in entries:
    with CC.internal():
        found = sorted(CC._dir(cache, e.key).glob("*.json"))
    if not found:
        why["no record under its key"] += 1
        continue
    reasons = []
    for p in found:
        try:
            rec = CC.read(p)
            if rec.get("key") != e.key:
                reasons.append("the record in its directory holds another key")
                continue
            reads = rec["reads"]
            now = CC._digests(cli.REPO, {k: list(v) for k, v in reads.items()})
            if not UN._unchanged(reads, now):
                reasons.append("a read changed")
                for kind, v in reads.items():
                    if UN._unchanged({kind: v}, now):
                        continue
                    kinds[kind] += 1
                    nv = now.get(kind)
                    if isinstance(v, dict) and isinstance(nv, dict):
                        for n in sorted(set(v) | set(nv), key=str):
                            if differs(kind, v.get(n), nv.get(n), n in v, n in nv):
                                names[f"{kind}: {n}"] += 1
                                ex.setdefault(f"{kind}: {n}", [str(v.get(n, "<not in the record>"))[:300], str(nv.get(n, "<not there now>"))[:300]])
                    else:
                        names[f"{kind}: <whole>"] += 1
                        ex.setdefault(f"{kind}: <whole>", [str(v)[:300], str(nv)[:300]])
                continue
            got = rec["report"]
            if got.get("roots") != e.root_ids:
                reasons.append("its roots differ")
                ex.setdefault(f"roots: {e.name}", [str(got.get("roots"))[:300], str(e.root_ids)[:300]])
                continue
            if not isinstance(got.get("nodes"), dict):
                reasons.append("it holds no nodes")
                continue
            for fid in e.root_ids:
                got["nodes"][fid]["k"]
            reasons.append("counts")
        except Exception as x:  # noqa: BLE001
            reasons.append(f"unreadable: {type(x).__name__}")
    if "counts" in reasons:
        counted += 1
        by_kind[f"{e.kind}: counts"] += 1
    else:
        why[reasons[0]] += 1
        by_kind[f"{e.kind}: misses"] += 1
        agree = UN._lookup(cache, cli.REPO, e) is None
        if not agree:
            why["(this probe says a miss where units._lookup gives a hit)"] += 1
print("WHY", json.dumps({"root_patched": os.environ.get("WHY_ROOT", "1") != "0", "cwd": os.getcwd(), "bindings": len(entries),
                         "count": counted, "miss_by_condition": dict(why), "by_binding_kind": dict(by_kind),
                         "kinds_of_read_that_changed": dict(kinds), "seconds": round(time.time() - t, 1)}), flush=True)
for n, c in names.most_common(12):
    print("WHY read", json.dumps({"read": n, "records": c, "recorded": ex[n][0], "now": ex[n][1]}), flush=True)
for n, v in list(ex.items()):
    if n.startswith("roots: "):
        print("WHY roots", json.dumps({"binding": n[7:], "recorded": v[0], "now": v[1]}), flush=True)
        break
