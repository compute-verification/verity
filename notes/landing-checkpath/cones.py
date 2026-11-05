"""Per landing and per after-landing batch on `checked`: the Lean files changed, the cone that imports them, its task
seconds (the executor's lean-module times of s20261103T000340Z-aeb35c, a median for a module it didn't time) and its
critical path, the longest import chain through the cone."""
import json, subprocess, sys, collections, statistics
from pathlib import Path

REPO = "/agent/repos/verity"
g = json.load(open("/tmp/cp/leangraph.json"))
sec = {m: v[0] for m, v in json.load(open("/tmp/cp/modsec.json")).items()}
MED = statistics.median(sec.values())
rdeps = collections.defaultdict(set)
for f, ds in g.items():
    for d in ds:
        rdeps[d].add(f)


def cost(f):
    return sec.get(f, MED)


def cone(changed):
    seen, stack = set(), [c for c in changed if c in g]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(rdeps[x])
    return seen


def critical(c):
    memo = {}

    def finish(f):
        if f in memo:
            return memo[f]
        memo[f] = 0
        memo[f] = cost(f) + max((finish(d) for d in g[f] if d in c), default=0)
        return memo[f]

    sys.setrecursionlimit(100000)
    return max((finish(f) for f in c), default=0)


def changed(a, b):
    out = subprocess.check_output(["git", "-C", REPO, "diff", "--name-only", a, b], text=True).split()
    return [f for f in out if f.endswith(".lean")]


def row(a, b, label):
    ch = changed(a, b)
    c = cone(ch)
    proofs = [f for f in c if f.startswith("Security/Proofs/")]
    return {"label": label, "from": a[:10], "to": b[:10], "lean_changed": len(ch), "changed_in_graph": len([x for x in ch if x in g]),
            "cone": len(c), "cone_proofs": len(proofs), "cone_task_s": round(sum(cost(f) for f in c)),
            "changed_task_s": round(sum(cost(f) for f in ch if f in g)), "critical_s": round(critical(c)),
            "critical_changed_only_s": round(max((cost(f) for f in ch if f in g), default=0))}


if __name__ == "__main__":
    fp = subprocess.check_output(["git", "-C", REPO, "rev-list", "--first-parent", "-80", "origin/checked"], text=True).split()
    rows = [row(p, c, "landing") for c, p in zip(fp, fp[1:])]
    tips = sys.argv[1:]
    rows += [row(a, b, "batch") for a, b in zip(tips, tips[1:])]
    json.dump(rows, open("/tmp/cp/cones.json", "w"), indent=0)
    for r in rows:
        print(r)
