#!/usr/bin/env python3
"""Part 1 at a commit, exact spellings: every old word of every word map (maps_words.py), whether a Glossary row names it,
its exact sites now, its new spelling's exact sites now, the verdict, and the three side tables.

  python3 old_words_v2.py PAIRS.tsv FOLDED_OLD_WORDS.tsv OUT_DIR [REF]

A site is the exact spelling, case kept, whole identifier or whole word (exactcount.py).  A word a map renamed only in listed
files ([[words_in]]) is counted in those files, followed through the renames of git when a file moved; the new spelling is
counted in the same files.  Verdict: GONE (the old spelling has no site), MAP DID NOT RUN (it has sites and the new spelling
has none), BOTH LIVE.  The folded count of the first pass stays in its own column and is not a count.
Writes old_words.v2.tsv, maps_not_run.tsv, both_live_sites.tsv, retired_by_row.txt."""
from __future__ import annotations

import collections
import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exactcount as X
import oldmeta
import scan

pairs_path, folded_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
ref = sys.argv[4] if len(sys.argv) > 4 else "HEAD"
out.mkdir(parents=True, exist_ok=True)
BASE = "c41a7b5a6d5c59bb95363a01770633933a83c7c5"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, errors="replace").stdout


m = scan.lint(); owner = scan.owners()
terms_by = m.terms()[1]
allowed = m.allowed()
rows = list(csv.DictReader(open(pairs_path), delimiter="\t"))
by_old = collections.defaultdict(list)
for r in rows:
    if r["old"] != r["new"]:
        by_old[r["old"]].append(r)
retired = sorted(o for o in by_old if m.sites_of(o, terms_by, "x.md"))
Path(out / "retired_by_row.txt").write_text("\n".join(retired) + "\n")
olds = [o for o in by_old if o not in set(retired)]
print(len(by_old), "old spellings,", len(retired), "retired by a lint-readable row,", len(olds), "left", file=sys.stderr)

tree = X.Tree(ref)
tracked = {p for p, _, _ in tree.files}
moved: dict[str, str | None] = {}


def resolve(f: str) -> str | None:
    if f in tracked:
        return f
    if f in moved:
        return moved[f]
    cur = f
    for _ in range(6):
        c = git("log", "-1", "--format=%H", "--diff-filter=DR", "-M", "--", cur).strip()
        nxt = None
        if c:
            for line in git("show", "-M", "--name-status", "--format=", c).splitlines():
                p = line.split("\t")
                if p[0].startswith("R") and len(p) == 3 and p[1] == cur:
                    nxt = p[2]
        if not nxt:
            break
        cur = nxt
        if cur in tracked:
            moved[f] = cur
            return cur
    moved[f] = None
    return None


def scope_of(o: str):
    ps = by_old[o]
    if any(p["scope"] == "words" for p in ps):
        return None
    files = {f for p in ps for f in p["scope"][len("words_in:"):].split(";") if f}
    return {r for r in (resolve(f) for f in files) if r}


def news_of(o: str):
    return sorted({n.strip() for p in by_old[o] for n in p["new"].split("|") if n.strip()})


scopes = {o: scope_of(o) for o in olds}
want = set(olds) | {n for o in olds for n in news_of(o)}
cnt = X.count(tree, want)


def total(s: str, scope):
    c = cnt[s]
    return sum(c.values()) if scope is None else sum(n for p, n in c.items() if p in scope)


folded = {}
if folded_path != "-" and Path(folded_path).exists():
    for r in csv.DictReader(open(folded_path), delimiter="\t"):
        folded[r["old"]] = r["sites"]

JUDGED = oldmeta.JUDGED
table = []
for o in olds:
    sc = scopes[o]
    news = news_of(o)
    old_n = total(o, sc)
    new_n = sum(total(n, sc) for n in news)
    verdict = "GONE" if old_n == 0 else ("MAP DID NOT RUN" if new_n == 0 else "BOTH LIVE")
    own = collections.Counter()
    for p, n in cnt[o].items():
        if sc is None or p in sc:
            own[owner(p) or "(none)"] += n
    ps = by_old[o]
    named = oldmeta.names_it(o)
    table.append(dict(
        old=o, new=" | ".join(news), map=" | ".join(dict.fromkeys(p["map"] for p in ps)),
        commit=" | ".join(dict.fromkeys(p["commit"] for p in ps)),
        applies="whole tree" if sc is None else "only in " + "; ".join(sorted(sc))[:160],
        old_sites=old_n, new_sites_now=new_n, verdict=verdict,
        old_files=sum(1 for p in cnt[o] if sc is None or p in sc),
        old_sites_tree_wide=sum(cnt[o].values()),
        owner_lanes=", ".join(f"{k}:{v}" for k, v in own.most_common(8)),
        folded_sites_not_a_count=folded.get(o, ""),
        glossary_names_it=named or "no", part_named=oldmeta.part_named(o),
        other_live_sense=oldmeta.live_sense(o, ps, sc is not None)))
table.sort(key=lambda r: (-r["old_sites"], r["old"]))
cols = list(table[0])
with open(out / "old_words.v2.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(table)
print(collections.Counter(r["verdict"] for r in table), file=sys.stderr)


def commit_info(map_name: str):
    h = git("log", "--full-history", "--diff-filter=A", "--format=%H %cI", "--", f"tools/move/{map_name}").split("\n")
    h = [x for x in h if x.strip()]
    if not h:
        return "", "", ""
    c, d = h[-1].split()
    return c[:12], d[:10], pr_of(c)


FIRST = git("rev-list", "--first-parent", BASE).split()
_pr = {}


def pr_of(commit):
    if commit in _pr:
        return _pr[commit]

    def has(i):
        return subprocess.run(["git", "merge-base", "--is-ancestor", commit, FIRST[i]]).returncode == 0
    if not has(0):
        _pr[commit] = "?"
        return "?"
    lo, hi = 0, len(FIRST) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if has(mid):
            lo = mid
        else:
            hi = mid - 1
    s = git("log", "-1", "--format=%s", FIRST[lo])
    _pr[commit] = (re.search(r"Queue: (#\d+)", s) or re.search(r"(#\d+)", s) or [None, "direct:" + FIRST[lo][:12]])[1]
    return _pr[commit]


per_map = collections.defaultdict(lambda: dict(words=0, dnr=0, sites=0, own=collections.Counter(), live=0))
for r in table:
    if r["verdict"] == "GONE":
        continue
    for mp in r["map"].split(" | "):
        d = per_map[mp]
        d["words"] += 1
        if r["verdict"] == "MAP DID NOT RUN":
            d["dnr"] += 1
            d["sites"] += r["old_sites"]
            for part in r["owner_lanes"].split(", "):
                if ":" in part:
                    k, v = part.rsplit(":", 1)
                    d["own"][k] += int(v)
mnr = []
for mp, d in per_map.items():
    if not d["dnr"]:
        continue
    c, dt, pr = commit_info(mp)
    mnr.append([mp, c, dt, pr, d["dnr"], d["words"], d["sites"], ", ".join(f"{k}:{v}" for k, v in d["own"].most_common(8)),
                "yes" if d["dnr"] == d["words"] else "no", "yes" if (Path("tools/move") / mp).exists() else "no"])
mnr.sort(key=lambda r: -r[6])
with open(out / "maps_not_run.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["map", "adding_commit", "date", "pr", "words_did_not_run", "words_with_sites", "exact_old_sites", "owner_lanes",
                "every_live_word_did_not_run", "map_file_in_tools_move_now"])
    w.writerows(mnr)

lint_terms = m.terms()[1]
with open(out / "both_live_sites.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["old", "file", "line", "text", "owner", "lint_reads_this_line", "lint_entry_for_file", "reading"])
    nrows = 0
    for r in table:
        if r["verdict"] != "BOTH LIVE":
            continue
        o = r["old"]; sc = scopes[o]
        paths = [p for p in cnt[o] if sc is None or p in sc]
        for path, ln, text in X.sites(tree, o, paths):
            reads = "yes" if m.sites_of(text, lint_terms, path) else "no"
            entry = f"allows {allowed.get(path, 0)}" if path in allowed else "no entry"
            w.writerow([o, path, ln, text.strip()[:200], owner(path) or "(none)", reads, entry, oldmeta.reading(path, text)])
            nrows += 1
print(nrows, "both-live sites", file=sys.stderr)
