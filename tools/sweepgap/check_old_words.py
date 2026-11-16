#!/usr/bin/env python3
"""The three verdicts on the first pass's table (old_words.tsv): did each word map run on REF?

Reads the rows whose other_live_sense starts with "no sign" (the rows top's check read), counts the old word and the
new spelling(s) in the tracked text files of REF outside tools/move/ and GLOSSARY.md, and prints the tally.
Default: top's reading (substring, as `str.count`), which must reproduce top_new_side.tsv; `--exact`: the exact
spelling (case kept, no letter, digit or underscore touching it).

    python3 check_old_words.py TABLE.tsv REF OUT.tsv [--exact] [--against top_new_side.tsv] [--checkout DIR]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exactcount as X

GONE = "GONE: the old word has no site now"
DID_NOT_RUN = "MAP DID NOT RUN: the new spelling has no site, the old word is the live name"
BOTH = "both live"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("ref"); ap.add_argument("out")
    ap.add_argument("--exact", action="store_true"); ap.add_argument("--against"); ap.add_argument("--checkout", default=".")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.table), delimiter="\t") if r["other_live_sense"].startswith("no sign")]
    news = {n.strip() for r in rows for n in r["new"].split("|") if n.strip()}
    olds = {r["old"] for r in rows}
    tree = X.Tree(a.ref, a.checkout)
    cnt = X.count(tree, news | olds, substring=not a.exact)
    total = {s: sum(c.values()) for s, c in cnt.items()}
    tally: Counter[str] = Counter()
    res = []
    for r in rows:
        new = [n.strip() for n in r["new"].split("|") if n.strip()]
        new_now = sum(total[n] for n in set(new))
        old_now = total[r["old"]]
        v = GONE if old_now == 0 else (DID_NOT_RUN if new_now == 0 else BOTH)
        tally[v] += 1
        res.append([r["old"], " | ".join(new), old_now, new_now, r["sites"], r["map"], v])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["old", "new", "old_sites_now", "new_sites_now", "worker_sites", "map", "verdict"])
        w.writerows(res)
    print(f"{a.ref}: {len(rows)} rows read as 'no sign' ({'exact' if a.exact else 'substring'})")
    for k, n in tally.most_common():
        print(f"{n:5d}  {k}")
    if a.against:
        theirs = {r["old"]: r for r in csv.DictReader(open(a.against), delimiter="\t")}
        diff = [(r[0], r[2], theirs[r[0]]["old_sites_now"], r[3], theirs[r[0]]["new_sites_now"]) for r in res
                if r[0] in theirs and (str(r[2]) != theirs[r[0]]["old_sites_now"] or str(r[3]) != theirs[r[0]]["new_sites_now"]
                                       or r[6] != theirs[r[0]]["verdict"])]
        print(f"against {a.against}: {len(res)} mine, {len(theirs)} theirs, {len(diff)} differ")
        for d in diff[:20]:
            print("  differs:", d)


if __name__ == "__main__":
    main()
