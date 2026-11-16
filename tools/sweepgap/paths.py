#!/usr/bin/env python3
"""Task 3a/3b: names the lint does not read.  Run from the checkout root:  python3 paths.py OUT_DIR
  path_names.tsv   each tracked file or directory component that the lint's matcher (GLOSSARY.md's rows) or one of today's
                   words (WORDS2) finds in its name: component, term, files under it, owner lane
  skipped_areas.tsv  sites of the rows' old names in what the lint does not read (SKIP, FIXTURE_DIRS, CHANGELOG, file types
                   it leaves out), by file: sites, spellings"""
import collections, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan as S
m = S.lint(); owner = S.owners(); by = m.terms()[1]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
files = [f for f in subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.decode().split("\0") if f]
WORDS2 = {  # today's words that the table does not list (sweep README), as they stand in a name
    "certifier": re.compile(r"certifier", re.I), "memory challenger": re.compile(r"memory[ _-]?challenger", re.I),
    "proof table": re.compile(r"proof[ _-]?table", re.I), "row/cell": re.compile(r"(?<![a-z])(rows?|cells?)(?![a-z])", re.I),
    "unit": re.compile(r"(?<![a-z])units?(?![a-z])", re.I),
}
comp = collections.defaultdict(lambda: [set(), set()])   # (component, term) -> [paths, owners]
for f in files:
    parts = f.split("/")
    for i, c in enumerate(parts):
        text = re.sub(r"\.[a-z0-9]+$", "", c) if i == len(parts) - 1 else c
        path = "/".join(parts[:i + 1]) + ("" if i == len(parts) - 1 else "/")
        for line, what in m.sites_of(text.replace("_", " ").replace("-", " "), by, "x.md"):
            mo = re.match(r"'(.+?)' is retired; say (.*) \(GLOSSARY\.md\)$", what, re.S)
            comp[(path, "row:" + " ".join(mo.group(1).split()))][0].add(f)
        for k, rx in WORDS2.items():
            if rx.search(text if k != "unit" else re.sub(r"(?<![a-z])unit[_-]?tests?", "", text, flags=re.I)):
                comp[(path, "today:" + k)][0].add(f)
with open(out / "path_names.tsv", "w") as fh:
    fh.write("path\tkind\tterm\tfiles_under\towner\n")
    for (p, t), (fs, _) in sorted(comp.items()):
        fh.write(f"{p}\t{'dir' if p.endswith('/') else 'file'}\t{t}\t{len(fs)}\t{owner(p.rstrip('/'))}\n")
print(len(comp), "path components")

# 3b: what the lint does not read
skip_rows, byext = [], collections.defaultdict(lambda: [0, 0])
def why_skipped(f):
    for s in m.SKIP:
        if f == s or s.endswith("/") and f.startswith(s):
            return "SKIP " + s
    if any(d in m.FIXTURE_DIRS for d in f.split("/")[:-1]):
        return "fixture dir"
    if f.rsplit("/", 1)[-1] in m.HISTORY:
        return "history file"
    if not f.endswith(m.PY + m.LEAN + m.MARKDOWN + m.HASH + m.SLASH):
        return "file type " + (f.rsplit(".", 1)[-1] if "." in f.rsplit("/", 1)[-1] else "(none)")
    return None
for f in files:
    why = why_skipped(f)
    if not why:
        continue
    p = Path(f)
    try:
        if p.stat().st_size > 3_000_000:
            continue
        t = p.read_text(encoding="utf-8")
    except Exception:
        continue
    sites = m.sites_of(t, by, f if f not in m.HELD else "x.md")
    if not sites:
        continue
    olds = collections.Counter(" ".join(re.match(r"(['\"])(.+?)\1 is retired", w).group(2).split()) for _, w in sites)
    skip_rows.append((why, f, len(sites), ", ".join(f"{k}:{v}" for k, v in olds.most_common(5)), owner(f)))
with open(out / "skipped_areas.tsv", "w") as fh:
    fh.write("why_the_lint_skips_it\tfile\tsites\ttop_old_names\towner\n")
    for r in sorted(skip_rows, key=lambda r: (r[0], -r[2])):
        fh.write("\t".join(map(str, r)) + "\n")
agg = collections.defaultdict(lambda: [0, 0])
for why, f, n, *_ in skip_rows:
    agg[why][0] += 1; agg[why][1] += n
with open(out / "skipped_areas_summary.tsv", "w") as fh:
    fh.write("why\tfiles\tsites\n")
    for k, (a, b) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        fh.write(f"{k}\t{a}\t{b}\n")
print(len(skip_rows), "skipped files with sites;", sum(r[2] for r in skip_rows), "sites")
