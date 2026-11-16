#!/usr/bin/env python3
"""Open ten sites of each BOTH LIVE word with over 50 old sites and say what holds the old spelling there.

  python3 read_sites.py PAIRS.tsv OLD_WORDS.v2.tsv BOTH_LIVE_SITES.tsv OUT_SITES.tsv OUT_SUMMARY.tsv [REF]

The ten are spread evenly over the word's sites (sorted by file, line).  A site is held when it is a stored record (the
store's moves.json, a lock record, a fixture), a path that still exists at REF and names a directory or file by the old
spelling, or a line one of the word's maps' [keep] patterns matches; else it is not held.  Each opened site's `reading`
starts with `opened:`; the others keep the first guess."""
from __future__ import annotations

import collections
import fnmatch
import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oldmeta
import warnings

warnings.simplefilter('ignore')

pairs_path, words_path, sites_path, out_sites, out_sum = sys.argv[1:6]
ref = sys.argv[6] if len(sys.argv) > 6 else "c41a7b5a6d5c59bb95363a01770633933a83c7c5"
csv.field_size_limit(1 << 28)

base_files = set(subprocess.run(["git", "ls-tree", "-r", "--name-only", ref], capture_output=True, text=True).stdout.split("\n"))
base_dirs = {"/".join(f.split("/")[:i]) for f in base_files for i in range(1, len(f.split("/")))}

pairs = collections.defaultdict(list)
for r in csv.DictReader(open(pairs_path), delimiter="\t"):
    pairs[r["old"]].append(r)
words = {r["old"]: r for r in csv.DictReader(open(words_path), delimiter="\t")}
sites = list(csv.DictReader(open(sites_path), delimiter="\t"))
by = collections.defaultdict(list)
for i, s in enumerate(sites):
    by[s["old"]].append(i)

PATHISH = re.compile(r"[\w.\-]+(?:/[\w.\-]+)+")


def compile_keeps(old: str):
    out = []
    for p in pairs[old]:
        for pat in oldmeta.keeps_of(p["map"], p["commit"]):
            try:
                out.append((p["map"], "[keep] pattern " + pat[:70], re.compile(pat), None))
            except re.error:
                pass
        for g, w in oldmeta.keeps_in_of(p["map"], p["commit"]):
            out.append((p["map"], f"[keep_in] {g}: {w}", re.compile(re.escape(w)), g))
    return out


def read(old: str, s: dict, keeps) -> tuple[str, str]:
    f, text = s["file"], s["text"]
    if f.endswith("moves.json"):
        return "held", "the store's moves.json records the move between the old and the new path (tools/research/src/research/store/moves.json)"
    if f.endswith("lean-audit.json"):
        return "held", f"a lock record: {f} names the declaration or file by its old spelling"
    if "/fixtures/" in f or f.endswith((".jsonl", ".bin", ".lock")):
        return "held", f"a fixture or lock file ({f}) that a reader reads byte for byte"
    if f.endswith("registry.json"):
        paths = [p for p in PATHISH.findall(text) if old.lower() in p.lower()]
        live = [p for p in paths if p in base_files or p in base_dirs]
        if live:
            return "held", f"the registry names {live[0]}, which still exists at the base"
        return "not held", f"the registry names {paths[0] if paths else 'a path'}, which does not exist at the base: a stale entry"
    for mp, what, rx, glob in keeps:
        if rx.search(text) and (glob is None or fnmatch.fnmatch(f, glob)):
            return "held", f"{mp}'s {what} matches the line"
    if "/" in old:
        live = [p for p in PATHISH.findall(text) if old in p and (p in base_files or p in base_dirs)]
        if live:
            return "held", f"{live[0]} still exists at the base, and the line names it"
        return "not held", "no such path exists at the base; the line is prose or code that names the old path"
    kind = "a Lean line" if f.endswith(".lean") else "a Python line" if f.endswith(".py") else "a Markdown or text line" if f.endswith((".md", ".txt")) else "a line of " + (Path(f).suffix or "text")
    maps = ", ".join(sorted({p["map"] for p in pairs[old]}))[:90]
    return "not held", f"{kind}; no [keep] pattern of {maps} matches and no stored record holds it"


summary = []
for old, idxs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    if len(idxs) <= 50:
        continue
    n = len(idxs)
    pick = sorted({idxs[int(i * (n - 1) / 9)] for i in range(10)})
    j = 0
    while len(pick) < 10:
        if idxs[j] not in pick:
            pick.append(idxs[j])
        j += 1
    keeps = compile_keeps(old)
    held = collections.Counter()
    for i in pick:
        verdict, why = read(old, sites[i], keeps)
        sites[i]["reading"] = f"opened: {verdict}: {why}"
        held[(verdict, why.split(" (")[0].split(":")[0][:60] if verdict == "held" else "")] += 1
    h = sum(c for (v, _), c in held.items() if v == "held")
    what = "; ".join(sorted({w for (v, w) in held if v == "held"}))
    summary.append([old, len(idxs), 10, h, 10 - h, what])

with open(out_sites, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(sites[0].keys()), delimiter="\t")
    w.writeheader()
    w.writerows(sites)
with open(out_sum, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["old", "sites", "opened", "held", "not_held", "what_holds"])
    w.writerows(summary)
for r in summary:
    print(*r, sep="\t")
