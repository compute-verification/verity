"""cov-difr: program.json diff of two runs (Definition sets, root body sampler callees, digests).  Usage: program_diff.py A_DIR B_DIR."""
import collections
import json
import re
import sys

A = json.load(open(f"{sys.argv[1]}/program.json"))
B = json.load(open(f"{sys.argv[2]}/program.json"))
na, nb = sys.argv[1].rstrip("/").split("/")[-1], sys.argv[2].rstrip("/").split("/")[-1]
pa, pb = A["program"], B["program"]
da, db = set(pa["definitions"]), set(pb["definitions"])
print(f"# program.json diff: {na} vs {nb}\n")
print("| | " + na + " | " + nb + " |\n|---|---|---|")
for k in ("profile", "digest", "n_instances", "n_input_nodes"):
    print(f"| {k} | {str(A[k])[:20]} | {str(B[k])[:20]} |")
print(f"| definitions | {len(da)} | {len(db)} |")
print(f"| callees | {len(pa['callees'])} | {len(pb['callees'])} |")
print(f"| types | {len(pa['types'])} | {len(pb['types'])} |")
print(f"| program_fn | {pa['program_fn']} | {pb['program_fn']} |")
print(f"| root | {pa['root']} | {pb['root']} |")
print(f"\n## Definitions only in {na} ({len(da - db)})\n")
for s in sorted(da - db):
    print("-", s)
print(f"\n## Definitions only in {nb} ({len(db - da)})\n")
for s in sorted(db - da):
    print("-", s)
shared = da & db
same = [s for s in shared if json.dumps(pa["definitions"][s], sort_keys=True) == json.dumps(pb["definitions"][s], sort_keys=True)]
print(f"\n## Shared definitions: {len(shared)}; body-identical: {len(same)}/{len(shared)}")
diff_shared = sorted(shared - set(same))
print("shared-but-different bodies:", diff_shared[:30])


def tally(body):
    t = collections.Counter()
    for m in re.findall(r"(TokenSelect_v1\{[^}]*\}|GumbelArgmax[A-Za-z0-9]*_v1\{[^}]*\})", json.dumps(body)):
        t[m] += 1
    return t


ka = pa["definitions"].get(pa["root"]) or {}
kb = pb["definitions"].get(pb["root"]) or {}
print(f"\nroot body json chars: {len(json.dumps(ka))} vs {len(json.dumps(kb))}")
print(f"sampler callee occurrences in the whole program -- {na}:", dict(sorted(tally(pa).items())))
print(f"sampler callee occurrences in the whole program -- {nb}:", dict(sorted(tally(pb).items())))
# the callee lists
ca, cb = collections.Counter(pa["callees"]), collections.Counter(pb["callees"])
print(f"\ncallees only in {na}:", sorted((ca - cb).elements())[:40])
print(f"callees only in {nb}:", sorted((cb - ca).elements())[:40])
