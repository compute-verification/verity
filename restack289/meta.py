#!/usr/bin/env python3
"""VLLM-289 scratch: our open pull requests with their queue state, candidates, stacks, and what `git merge BASE` does to
each, read on a VM with git alone -> OUT (default /tmp/rs/meta.json).

    meta.py BASE [OUT] [LABEL=COMMIT...]

BASE is the commit the layout is staged on. For each head (and each LABEL=COMMIT, a mend): whether BASE holds it, the files
`git merge BASE` stops on (`git merge-tree`, with the clone's merge drivers), and the `Move:` commits BASE has and the head
lacks, which `restack.py` would take for moves to cross where the merge is skipped.
"""
import json
import re
import subprocess
import sys

sys.path.insert(0, "tools/move")
import restack as RS  # noqa: E402

BASE = RS.rev(sys.argv[1], None)
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/rs/meta.json"
EXTRA = [a.split("=", 1) for a in sys.argv[3:]]


def sh(*a, ok=(0,)):
    p = subprocess.run(a, capture_output=True, text=True)
    if p.returncode not in ok:
        raise SystemExit(f"{a[:4]}: exit {p.returncode}: {p.stderr[-300:]}")
    return p.stdout


_ANC: dict = {}


def anc(a, b):
    if (a, b) not in _ANC:
        _ANC[a, b] = subprocess.run(["git", "merge-base", "--is-ancestor", a, b]).returncode == 0
    return _ANC[a, b]


def crossing(head):
    """What the day's first command does at ``head``, and the moves BASE holds that it lacks."""
    if anc(head, BASE):
        return {"in_base": True, "merge_base": [], "lacks_moves": []}
    _tree, conf = RS.merge_trees(head, BASE, None)
    return {"in_base": False, "merge_base": conf, "lacks_moves": [c[:12] for c, _s, _a in RS.moves_on(BASE, head, None)]}


prs = json.loads(sh("gh", "pr", "list", "--state", "open", "--limit", "500", "--json",
                    "number,title,headRefName,headRefOid,baseRefName,body,updatedAt,isDraft"))
ours = [p for p in prs if re.search(r"\bLane vllm\b", p.get("body") or "") or p["headRefName"].startswith("cursor/vllm-")]
by_branch = {p["headRefName"]: p["number"] for p in prs}
queue = {r["pr"]: r for r in json.loads(sh("research", "queue", "status", "--json"))}
unreviewed = {r["pr"]: r for r in json.loads(sh("research", "queue", "review", "--json"))}

refs = [line.split("\t") for line in sh("git", "ls-remote", "origin", "refs/heads/queue/*", "refs/heads/main").splitlines()]
tips = {ref.removeprefix("refs/heads/"): sha for sha, ref in refs}
names = sorted(n for n in tips if n.startswith("queue/"))
sh("git", "fetch", "-q", "origin", "main", *names)
missing = [p["headRefOid"] for p in ours
           if subprocess.run(["git", "cat-file", "-e", p["headRefOid"] + "^{commit}"], capture_output=True).returncode]
if missing:
    sh("git", "fetch", "-q", "origin", *missing)
main = tips["main"]
cands = {}
for n in names:
    subjects = sh("git", "log", "--first-parent", "--format=%s", f"{main}..{tips[n]}").splitlines()
    cands[n.removeprefix("queue/")] = [int(m.group(1)) for s in subjects if (m := re.match(r"Queue: #(\d+)\b", s))]
head_of = {p["number"]: p["headRefOid"] for p in ours}

rows = []
for p in sorted(ours, key=lambda p: p["number"]):
    n, h = p["number"], p["headRefOid"]
    q = queue.get(n, {})
    under = [o for o, oh in head_of.items() if o != n and oh != h and anc(oh, h)]
    near = [u for u in under if not any(v != u and anc(head_of[u], head_of[v]) for v in under)]
    base_pr = by_branch.get(p["baseRefName"]) if p["baseRefName"] != "main" else None
    marked = q.get("state") in ("ready", "refused") and "not marked ready" not in (q.get("why") or "")
    rows.append({"pr": n, "branch": p["headRefName"], "head": h, "base": p["baseRefName"], "base_pr": base_pr,
                 "draft": p["isDraft"], "marked": marked, "state": q.get("state"), "why": q.get("why"),
                 "verdict": ("yes" if n not in unreviewed else "no") if marked else "-",
                 "candidates": [c for c, ns in cands.items() if n in ns], "on_main": anc(h, main),
                 "stacked_on": sorted(set(near) | ({base_pr} if base_pr else set())),
                 "behind": int(sh("git", "rev-list", "--count", "--first-parent", f"{h}..{main}")),
                 "title": p["title"], "updated": p["updatedAt"], **crossing(h)})
extra = {label: {"head": RS.rev(c, None), **crossing(RS.rev(c, None))} for label, c in EXTRA}
json.dump({"main": main, "base": BASE, "candidates": {c: {"tip": tips["queue/" + c], "prs": ns} for c, ns in cands.items()},
           "rows": rows, "extra": extra}, open(OUT, "w"), indent=1)
print(len(rows), "of", len(prs), "open; main", main[:12], "base", BASE[:12], "candidates", {c: len(v) for c, v in cands.items()})
