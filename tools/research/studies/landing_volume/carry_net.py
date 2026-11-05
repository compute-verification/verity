"""For each re-head that brought in other PRs' heads (carry_in.py's CSV): whether the PR's own edit, net of what it brought in,
is unchanged, i.e. every file whose zero-context edit differs between the old and the new head is a file one of the brought-in
PRs changes, or a generated file (reheads.GENERATED; none under STRICT=1, since a reviewer reads a lock's merge too).  Then a
verdict on the old head judged the same own change.

    [STRICT=1] python3 carry_net.py REHEADS_CARRY_CSV LABELS_JSON REPO OUT_CSV OPEN_JSON CLOSED_JSON...
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reheads import GENERATED, patch_ids  # noqa: E402


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="replace").stdout


def main(carry_csv, labels_json, repo, out_csv, *prs_json):
    labels = json.load(open(labels_json))
    full: dict[tuple[int, str], str] = {}
    last: dict[int, str] = {}
    for n, labs in labels.items():
        for x in labs:
            h = x["target"].split("@", 1)[1]
            full[(int(n), h[:12])] = h
    for f in prs_json:
        for x in json.load(open(f)):
            full[(x["number"], x["headRefOid"][:12])] = x["headRefOid"]
    rows = list(csv.DictReader(open(carry_csv)))
    # the head of each brought-in PR that the new head contains: the newest of its known heads in the new head's history
    heads_of: dict[int, list[str]] = {}
    for (n, _), h in full.items():
        heads_of.setdefault(n, []).append(h)

    def ck(ts):
        return g(repo, "rev-list", "-1", "--first-parent", f"--before={ts}", "origin/checked").strip() if ts else "origin/checked"

    def one(r):
        if not r["carried_in"] or r["kind"] in ("missing", "big", "sync", "sync+regen"):
            return {**r, "net": ""}
        n = int(r["pr"])
        a, b = full[(n, r["old"])], full[(n, r["new"])]
        ca, cb = ck(r["old_first_label"]), ck(r["new_first_label"])
        pa, pb = patch_ids(repo, a, ca), patch_ids(repo, b, cb)
        differ = {f for f in set(pa) | set(pb) if pa.get(f) != pb.get(f)}
        theirs: set[str] = set()
        for q in (int(x) for x in r["carried_in"].split(";")):
            for h in heads_of.get(q, ()):
                if not subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", h, b]).returncode:
                    mb = g(repo, "merge-base", cb, h).strip()
                    theirs |= set(g(repo, "diff", "--no-renames", "--name-only", mb, h).split())
        gen = () if os.environ.get("STRICT") == "1" else GENERATED
        mine = {f for f in differ - theirs if not any(f.endswith(s) or f"/{s}" in f for s in gen)}
        return {**r, "net": "unchanged" if not mine else "changed", "own_differ": len(mine)}

    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(one, rows))
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(Counter(r["net"] for r in out if r["net"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
