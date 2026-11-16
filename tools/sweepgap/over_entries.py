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
over = {f: v for f, v in where.items() if len(v) > have[f]}
pr_of = {}
def pr(commit):
    """The queue merge that brought ``commit`` to origin/main: the first-parent step git name-rev names it by."""
    if commit not in pr_of:
        name = git("name-rev", "--name-only", "--no-undefined", "--refs=refs/remotes/origin/main", commit).strip()
        mo = re.match(r"(?:remotes/)?origin/main(?:~(\d+))?(\^2.*)?$", name) if name else None
        if not mo:
            pr_of[commit] = "?"
        else:
            merge = git("rev-parse", f"origin/main~{mo.group(1) or 0}").strip()
            s = git("log", "-1", "--format=%s", merge)
            pr_of[commit] = (re.search(r"Queue: (#\d+)", s) or re.search(r"(#\d+)", s) or [None, "direct:" + merge[:12]])[1]
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
        rows.append((f, l, " ".join(old.split()), c[:12], pr(c) if c != "?" else "?", subprocess.run(["date", "-u", "-d", f"@{t}", "+%Y-%m-%dT%H:%MZ"], capture_output=True, text=True).stdout.strip(),
                     s[:90], f"{len(sites)} found, {have[f]} allowed", owner(f), entry[0][:12] + " " + entry[1] if entry else "no entry ever"))
with open(sys.argv[1], "w") as fh:
    fh.write("file\tline\told\tcommit\tpr\tcommit_date\tcommit_subject\tfound_allowed\towner\tentry_last_set\n")
    for r in rows: fh.write("\t".join(map(str, r)) + "\n")
print(len(rows), "sites in", len(over), "files")
