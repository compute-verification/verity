"""What each re-head after a ready mark or a land verdict changed: for each PR, its labelled heads in the order first labelled,
and for each consecutive pair (old, new) whether new descends from old, and whether the PR's own change is the same.

    python3 reheads.py LABELS_JSON REPO OUT_CSV OPEN_JSON CLOSED_JSON...

A head's own change is {(path, blob at the head)} over the paths it changes against its merge base with `main` at the new
head's time (approximated by `checked` now); two heads with the same own change differ only by what they merged from `main` or
`checked` (a sync), so a verdict on the old one judged exactly the change the new one brings.  `kind`:
  sync          new descends from old, the own change is identical: a merge of main (or of a landed base) with no edit;
  sync+resolve  new descends from old, the same paths, some blob differs only where the base moved too (conflict resolution
                or a regenerated lock/pin);
  edit          new descends from old and changes the PR's own content;
  rewrite       new doesn't descend from old (a rebase, squash or rebuild);
  missing       old isn't fetchable any more.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from contain import lane  # noqa: E402


def g(repo, *a, check=False):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="replace", check=check)


def own(repo: str, h: str, base: str) -> dict[str, str | None] | None:
    mb = g(repo, "merge-base", base, h).stdout.strip()
    if not mb:
        return None
    files = [f for f in g(repo, "diff", "--no-renames", "--name-only", mb, h).stdout.splitlines() if f]
    if not files:
        return {}
    blobs = {}
    for line in g(repo, "ls-tree", h, "--", *files).stdout.splitlines():
        meta, path = line.split("\t", 1)
        blobs[path] = meta.split()[2]
    return {f: blobs.get(f) for f in files}


#: files a sync regenerates rather than edits: the Lean locks and the generated pins
GENERATED = ("lean-audit.json", "pins", ".sha256", "uv.lock", "lake-manifest.json")


def patch_ids(repo: str, h: str, base: str) -> dict[str, str]:
    """{path: patch-id of its zero-context diff against the merge base}: equal across two heads when the PR's own edit of the
    file is the same whatever else its base changed in it."""
    mb = g(repo, "merge-base", base, h).stdout.strip()
    out = {}
    for f in g(repo, "diff", "--no-renames", "--name-only", mb, h).stdout.splitlines():
        d = g(repo, "diff", "--no-renames", "-U0", mb, h, "--", f).stdout
        pid = subprocess.run(["git", "-C", repo, "patch-id", "--stable"], input=d, capture_output=True, text=True,
                             errors="replace").stdout.split()
        out[f] = pid[0] if pid else ""
    return out


def main(labels_json: str, repo: str, out_csv: str, open_json: str, *closed: str) -> None:
    labels = {int(k): v for k, v in json.load(open(labels_json)).items()}
    prs = {}
    for f in (open_json, *closed):
        for x in json.load(open(f)):
            prs[x["number"]] = x
    todo, want = [], set()
    for n, labs in labels.items():
        if n not in prs:
            continue
        first: dict[str, str] = {}
        kinds: dict[str, set] = {}
        for x in sorted(labs, key=lambda x: x["ts"]):
            if x["key"] not in ("ready", "reviewed"):
                continue
            h = x["target"].split("@", 1)[1]
            first.setdefault(h, x["ts"])
            kinds.setdefault(h, set()).add("verdict" if x["key"] == "reviewed" and x["value"] == "land"
                                           else "mark" if x["key"] == "ready" and str(x["value"]).lower() == "true" else "other")
        hs = sorted(first, key=first.get)
        final = prs[n]["headRefOid"]
        if final not in first:
            hs.append(final)
            kinds[final] = set()
        for a, b in zip(hs, hs[1:]):
            todo.append((n, a, b, sorted(kinds.get(a, ())), first.get(a, ""), first.get(b, "")))
            want |= {a, b}
    have = {h for h in want if g(repo, "cat-file", "-e", f"{h}^{{commit}}").returncode == 0}
    missing = sorted(want - have)
    for i in range(0, len(missing), 200):
        g(repo, "fetch", "-q", "origin", *missing[i:i + 200])
    def at(ts: str) -> str:
        """`checked` as it stood at ``ts`` (now, for a head never labelled)."""
        if not ts:
            return g(repo, "rev-parse", "origin/checked").stdout.strip()
        return g(repo, "rev-list", "-1", "--first-parent", f"--before={ts}", "origin/checked").stdout.strip()

    def one(t):
        n, a, b, ka, ta, tb = t
        row = {"pr": n, "lane": lane(prs[n]), "outcome": "landed" if prs[n].get("mergedAt") else "closed" if prs[n].get("closedAt") else "open",
               "old": a[:12], "new": b[:12], "old_had": "+".join(ka), "old_first_label": ta, "new_first_label": tb}
        if any(g(repo, "cat-file", "-e", f"{h}^{{commit}}").returncode for h in (a, b)):
            return {**row, "kind": "missing"}
        desc = g(repo, "merge-base", "--is-ancestor", a, b).returncode == 0
        base = at(tb)
        oa, ob = own(repo, a, at(ta)), own(repo, b, base)
        if not oa and not ob:
            kind = "empty"
        elif not desc:
            kind = "rewrite" if oa != ob else "rewrite-same"
        elif oa == ob:
            kind = "sync"
        else:
            pa, pb = patch_ids(repo, a, at(ta)), patch_ids(repo, b, base)
            differ = sorted(f for f in set(pa) | set(pb) if pa.get(f) != pb.get(f))
            if not differ:
                kind = "sync"
            elif all(any(f.endswith(s) or f"/{s}" in f for s in GENERATED) for f in differ):
                kind = "sync+regen"
            elif set(pa) == set(pb):
                kind = "same-files-edit"
            else:
                kind = "edit"
            row["differ"] = ";".join(differ[:5])
        between = g(repo, "rev-list", "--count", "--no-merges", f"{a}..{b}", "--not", base).stdout.strip() if desc else ""
        return {**row, "kind": kind, "own_commits_between": between,
                "files_old": len(oa or {}), "files_new": len(ob or {})}

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, todo))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(len(rows), Counter(r["kind"] for r in rows))
    print(Counter((r["kind"], r["old_had"]) for r in rows).most_common(20))


if __name__ == "__main__":
    main(*sys.argv[1:])
