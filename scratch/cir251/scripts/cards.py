"""Three-way merge of a one-stage partition card by key: main's card, with each template this branch renamed (base -> ours,
paired by position among the entries that differ) renamed in place in ``templates`` and as a ``chunk_cap`` key, its cap
kept.  Refuses a template both sides changed.  Usage: cards.py BASE OURS THEIRS PATH...  (run in the worktree)"""
import json
import subprocess
import sys

base, ours, theirs, *paths = sys.argv[1:]


def load(rev: str, p: str) -> dict:
    return json.loads(subprocess.check_output(["git", "show", f"{rev}:{p}"]))


for p in paths:
    b, o, t = load(base, p), load(ours, p), load(theirs, p)
    gone = [x for x in b["templates"] if x not in o["templates"]]
    new = [x for x in o["templates"] if x not in b["templates"]]
    assert len(gone) == len(new), (p, gone, new)
    ren = dict(zip(gone, new))
    assert all(b["chunk_cap"].get(g) == o["chunk_cap"].get(n) for g, n in ren.items()), p
    assert {k: v for k, v in b.items() if k not in ("templates", "chunk_cap")} == \
        {k: v for k, v in o.items() if k not in ("templates", "chunk_cap")}, p
    missing = [g for g in ren if g not in t["templates"]]
    assert not missing, (p, "main changed these too", missing)
    t["templates"] = [ren.get(x, x) for x in t["templates"]]
    t["chunk_cap"] = {ren.get(k, k): v for k, v in t["chunk_cap"].items()}
    with open(p, "w") as f:
        f.write(json.dumps(t) + "\n")
    print(p, json.dumps(ren))
