#!/usr/bin/env python3
"""The sites of the retired-terms lint past their files' entries: for each file that holds more sites than
tests/lint/allowlists/repo_retired_terms.json allows, every site (line, old word) with the commit that added it
(git blame), the queue merge (pull request) that brought that commit to main, its date, and the owner lane.
Run from the checkout root:  python3 over_entries.py OUT.tsv     (about 40 s; reads origin/main's first-parent history)"""
import collections, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan as S
m = S.lint(); owner = S.owners()
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout
have = m.allowed(); where = collections.defaultdict(list)
for f, l, w in m.scan(): where[f].append((l, w))
over = {f: v for f, v in where.items() if len(v) > have[f] and not f.startswith('tools/sweepgap/')}
pr_of = {}
FIRST = git("rev-list", "--first-parent", "c41a7b5a6d5c59bb95363a01770633933a83c7c5").split()
def pr(commit):
    """The first-parent commit of origin/main that first holds ``commit`` (binary search: what an ancestor test says is
    monotone along the first-parent line), named by its queue number."""
    if commit not in pr_of:
        def has(i): return subprocess.run(["git", "merge-base", "--is-ancestor", commit, FIRST[i]]).returncode == 0
        lo, hi = 0, len(FIRST) - 1          # FIRST[0] is the newest; find the largest i with has(i)
        if not has(0):
            pr_of[commit] = "?"
        else:
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if has(mid): lo = mid
                else: hi = mid - 1
            s = git("log", "-1", "--format=%s", FIRST[lo])
            pr_of[commit] = (re.search(r"Queue: (#\d+)", s) or re.search(r"(#\d+)", s) or [None, "direct:" + FIRST[lo][:12]])[1]
    return pr_of[commit]
LIST = "tests/lint/allowlists/repo_retired_terms.json"
rows = []
for f, sites in sorted(over.items()):
    blame = {}
    cur = None
    for line in git("blame", "--line-porcelain", "HEAD", "--", f).split("\n"):
        mo = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line)
        if mo: cur = [mo.group(1), int(mo.group(2)), None, None]; blame[cur[1]] = cur
        elif cur and line.startswith("author-time "): cur[2] = int(line.split()[1])
        elif cur and line.startswith("summary "): cur[3] = line[8:]
    entry = git("log", "-1", "--format=%H %cI", "-G" + re.escape(f'"file": "{f}"'), "--", LIST).split()
    for l, w in sites:
        c, _, t, s = blame.get(l, ["?", l, 0, "?"])
        old = re.match(r"'(.+?)' is retired", w).group(1)
        when = subprocess.run(["date", "-u", "-d", f"@{t}", "+%Y-%m-%dT%H:%MZ"], capture_output=True, text=True).stdout.strip()
        after = "yes" if entry and when > subprocess.run(["date", "-u", "-d", entry[1], "+%Y-%m-%dT%H:%MZ"], capture_output=True, text=True).stdout.strip() else ("no" if entry else "no entry")
        rows.append((f, l, " ".join(old.split()), c[:12], pr(c) if c != "?" else "?", when,
                     s[:90], f"{len(sites)} found, {have[f]} allowed", owner(f), entry[0][:12] + " " + entry[1] if entry else "no entry ever", after))
with open(sys.argv[1], "w") as fh:
    fh.write("file\tline\told\tcommit\tpr\tcommit_date\tcommit_subject\tfound_allowed\towner\tentry_last_set\tadded_after_entry\n")
    for r in rows: fh.write("\t".join(map(str, r)) + "\n")
print(len(rows), "sites in", len(over), "files")
