"""Open PRs that are carried or superseded: by ancestry and by blob, against `main`, `checked`, `next` and each other.

    python3 contain.py OPEN_JSON REPO OUT_DIR

OPEN_JSON is `gh pr list --state open --json number,headRefName,headRefOid,body,createdAt,updatedAt,baseRefName,title`; REPO
has `refs/vol/pr/N` (each open PR's head) and origin's `main`, `checked`, `next` fetched.  Writes OUT_DIR/prs.json (one
record per PR) and OUT_DIR/pairs.json (each carrier relation found, with how it was found).

A PR's change is its files that differ between its head and its merge base with `checked` (no rename detection).  It is
*applied* on a commit B when every one of those files has on B the blob it has at the PR's head (absent on both, for a
deletion): what it brings is already there, by ancestry, a squash, a fold or a cherry-pick.  This is conservative: a carrier
that changed one of the files again doesn't count.  P *carries* Q when Q's head is an ancestor of P's head (ancestry), or
Q's change, not applied on `checked`, is applied on P's head (blob).  `git merge-tree` would judge more, but takes minutes
on a head thousands of commits behind `main`.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LANE_RE = re.compile(r"^[ \t>*_]*Lane[:\s]+`?([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)`?", re.M)
AGENT_RE = re.compile(r"\bbc-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
REF_RE = re.compile(r"#(\d{4,5})\b")
CARRY_RE = re.compile(r"(carr(?:y|ies|ied)|supersed\w*|replac\w*|fold\w*|subsum\w*|merges|includes|absorb\w*)[^.\n]{0,80}?#(\d{4,5})",
                      re.I)


def git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=check)


def out(repo: str, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def lane(p: dict) -> str:
    if m := LANE_RE.search(p.get("body") or ""):
        return m.group(1).rstrip(".")
    if m := AGENT_RE.search(p.get("body") or ""):
        return m.group(0)
    b = p["headRefName"]
    return "branch:" + (b.split("/", 1)[1].split("-")[0] if "/" in b else b)


def blobs(repo: str, commit: str) -> dict[str, str]:
    return {line.split("\t", 1)[1]: line.split()[2] for line in out(repo, "ls-tree", "-r", commit).splitlines()}


def main(open_json: str, repo: str, out_dir: str) -> None:
    prs = json.load(open(open_json))
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tips = {b: out(repo, "rev-parse", f"origin/{b}") for b in ("main", "checked", "next")}
    tip_blobs = {b: blobs(repo, c) for b, c in tips.items()}
    by_num = {p["number"]: p for p in prs}
    heads = {p["headRefOid"]: p["number"] for p in prs}
    members = {}
    for chunk in out(repo, "log", "--first-parent", "--format=%H%x00%B%x01", f"{tips['checked']}..{tips['next']}").split("\x01"):
        if not chunk.strip():
            continue
        c, body = chunk.strip().split("\x00", 1)
        pr = re.search(r"^PR: #(\d+)", body, re.M)
        hd = re.search(r"^Head: ([0-9a-f]{40})", body, re.M)
        if pr and hd:
            members[int(pr.group(1))] = {"commit": c, "head": hd.group(1)}

    def changed(p: dict) -> list[str]:
        mb = out(repo, "merge-base", tips["checked"], p["headRefOid"])
        return [f for f in out(repo, "diff", "--no-renames", "--name-only", mb, p["headRefOid"]).splitlines() if f]

    with ThreadPoolExecutor(16) as ex:
        files = dict(zip((p["number"] for p in prs), ex.map(changed, prs)))
    touched = set().union(*files.values())

    def one(p: dict) -> dict:
        n, h = p["number"], p["headRefOid"]
        body = p.get("body") or ""
        r = {"pr": n, "lane": lane(p), "head": h, "base": p["baseRefName"], "created": p["createdAt"], "updated": p["updatedAt"],
             "title": p.get("title", ""), "on_next": n in members, "next_head_current": members.get(n, {}).get("head") == h}
        for b, c in tips.items():
            r[f"ancestor_{b}"] = git(repo, "merge-base", "--is-ancestor", h, c, check=False).returncode == 0
        r["behind_checked"] = int(out(repo, "rev-list", "--count", f"{h}..{tips['checked']}") or 0)
        r["files"] = files[n]
        r["own_commits"] = int(out(repo, "rev-list", "--count", f"{tips['checked']}..{h}") or 0)
        r["contains_heads"] = sorted(heads[c] for c in out(repo, "rev-list", f"{tips['checked']}..{h}").splitlines()
                                     if c in heads and heads[c] != n)
        mine = blobs(repo, h)
        r["_blobs"] = {f: mine.get(f) for f in r["files"]}
        r["_tree"] = {f: mine.get(f) for f in touched if f in mine}
        for b in ("main", "checked", "next"):
            same = sum(tip_blobs[b].get(f) == mine.get(f) for f in r["files"])
            r[f"applied_{b}"] = bool(r["files"]) and same == len(r["files"])
            r[f"same_files_{b}"] = same
        r["mentions"] = sorted({int(x) for x in REF_RE.findall(body) if int(x) in by_num and int(x) != n})
        r["carry_mentions"] = sorted({int(m.group(2)) for m in CARRY_RE.finditer(body) if int(m.group(2)) != n})
        return r

    with ThreadPoolExecutor(16) as ex:
        recs = list(ex.map(one, prs))
    rec = {r["pr"]: r for r in recs}

    pairs = []
    for r in recs:
        for q in r["contains_heads"]:
            pairs.append({"carrier": r["pr"], "carried": q, "how": "ancestry"})
    anc = {(x["carrier"], x["carried"]) for x in pairs}
    for q in recs:
        if not q["files"] or q["applied_checked"] or q["ancestor_checked"]:
            continue
        for p in recs:
            if p["pr"] == q["pr"] or (p["pr"], q["pr"]) in anc:
                continue
            if all(p["_tree"].get(f) == b for f, b in q["_blobs"].items()):
                pairs.append({"carrier": p["pr"], "carried": q["pr"], "how": "blob"})
    for r in recs:
        r.pop("_tree")
        r.pop("_blobs")
    (od / "prs.json").write_text(json.dumps(recs, indent=1))
    (od / "pairs.json").write_text(json.dumps({"tips": tips, "members": members, "pairs": pairs}, indent=1))
    print(f"{len(recs)} PRs, {len(members)} next members, {len(pairs)} pairs")


if __name__ == "__main__":
    main(*sys.argv[1:4])
