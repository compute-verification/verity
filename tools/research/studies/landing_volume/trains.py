"""Landed PRs that another landed PR carried: each `Queue: #N` merge on `checked` since SINCE whose landed head (the merge's
second parent) is in a later landing's head history but was not on `checked` when that later PR was first marked. Such a
PR was its own member of `next` (admission, preflight, verdict, merge) though the carrier brought it anyway: what a lane
train (mark only the carrier) would have spent once.

    python3 trains.py LABELS_JSON REPO SINCE OUT_CSV PRS_JSON...
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from contain import lane

BIG = 5000


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="replace").stdout


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main(labels_json, repo, since, out_csv, *prs_json):
    labels = json.load(open(labels_json))
    meta: dict[int, dict] = {}
    for f in prs_json:
        for x in json.load(open(f)):
            meta.setdefault(x["number"], x)
    landed: dict[int, tuple[str, str]] = {}
    for line in g(repo, "log", "--first-parent", f"--since={since}", "--format=%H %P%x09%cI%x09%s", "origin/checked").splitlines():
        hp, t, s = line.split("\t", 2)
        m = re.match(r"Queue: #(\d+)", s)
        parents = hp.split()
        if m and len(parents) == 3:
            landed.setdefault(int(m.group(1)), (parents[2], t))
    by_head = {h: n for n, (h, _) in landed.items()}

    def first_mark(n: int, h: str) -> str | None:
        t = [x["ts"] for x in labels.get(str(n), []) if x["target"].endswith("@" + h) and x.get("key") == "ready"
             and x.get("value") is True]
        return min(t) if t else None

    def scan(n: int):
        h, t = landed[n]
        m = first_mark(n, h) or t
        ck = g(repo, "rev-list", "-1", "--first-parent", f"--before={m}", "origin/checked").strip()
        revs = g(repo, "rev-list", f"--max-count={BIG}", h, f"^{ck}").split()
        return n, m, len(revs), [by_head[r] for r in revs if r in by_head and by_head[r] != n]

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(scan, sorted(landed)))
    carried_by: dict[int, list[int]] = {}
    for q, _, size, ps in res:
        if size >= BIG:
            continue
        for p in ps:
            if ts(landed[p][1]) <= ts(landed[q][1]):
                carried_by.setdefault(p, []).append(q)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pr", "lane", "landed", "carriers", "first_carrier", "carrier_lane", "same_lane", "h_to_carrier_landing"])
        for p in sorted(landed):
            qs = sorted(carried_by.get(p, []), key=lambda q: landed[q][1])
            lp = lane(meta[p]) if p in meta else "?"
            if not qs:
                w.writerow([p, lp, landed[p][1], "", "", "", "", ""])
                continue
            q = qs[0]
            lq = lane(meta[q]) if q in meta else "?"
            dh = (ts(landed[q][1]) - ts(landed[p][1])).total_seconds() / 3600
            w.writerow([p, lp, landed[p][1], " ".join(map(str, qs)), q, lq, int(lp == lq), f"{dh:.2f}"])


if __name__ == "__main__":
    main(*sys.argv[1:])
