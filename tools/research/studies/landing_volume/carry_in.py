"""Which re-heads brought in another PR's head: for each (old, new) of reheads.py's CSV, the PRs (open or closed, by any head
the store has labels on or GitHub lists) whose head is in new's history but not old's, nor on `checked` at new's time.

    python3 carry_in.py REHEADS_CSV LABELS_JSON REPO OUT_CSV OPEN_JSON CLOSED_JSON...
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="replace").stdout


def main(reheads_csv, labels_json, repo, out_csv, *prs_json):
    labels = json.load(open(labels_json))
    heads: dict[str, int] = {}
    full: dict[tuple[int, str], str] = {}
    for n, labs in labels.items():
        for x in labs:
            h = x["target"].split("@", 1)[1]
            heads[h] = int(n)
            full[(int(n), h[:12])] = h
    for f in prs_json:
        for x in json.load(open(f)):
            heads.setdefault(x["headRefOid"], x["number"])
            full[(x["number"], x["headRefOid"][:12])] = x["headRefOid"]
    rows = list(csv.DictReader(open(reheads_csv)))

    def one(r):
        n = int(r["pr"])
        a, b = full.get((n, r["old"])), full.get((n, r["new"]))
        if not a or not b or r["kind"] == "missing":
            return {**r, "carried_in": ""}
        ck = g(repo, "rev-list", "-1", "--first-parent", f"--before={r['new_first_label']}", "origin/checked").strip() \
            if r["new_first_label"] else "origin/checked"
        got = {heads[c] for c in g(repo, "rev-list", b, f"^{a}", f"^{ck}").split() if c in heads and heads[c] != n}
        return {**r, "carried_in": ";".join(map(str, sorted(got)))}

    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(one, rows))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(Counter((r["kind"], bool(r["carried_in"])) for r in out))


if __name__ == "__main__":
    main(*sys.argv[1:])
