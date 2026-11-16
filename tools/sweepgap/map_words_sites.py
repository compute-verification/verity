#!/usr/bin/env python3
"""For each old word of pairs.tsv (maps_words.py): is it retired by a row of GLOSSARY.md (the lint's matcher)? If not,
its sites in today's tree (tracked text files under 1 MB, not tools/move/ nor GLOSSARY.md): whole (git grep -n -i -w) and
as one part of an identifier (snake, camel, capitals), by owner lane, three examples.
Run from the checkout root:  python3 map_words_sites.py PAIRS.tsv OUT_DIR"""
import collections, csv, importlib.util, json, re, subprocess, sys
from pathlib import Path
SWEEP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scan", SWEEP / "scan.py"); scan = importlib.util.module_from_spec(spec); spec.loader.exec_module(scan)
m = scan.lint(); owner = scan.owners()
pairs_path, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(pairs_path), delimiter="\t"))
by_old = collections.defaultdict(list)
for r in rows: by_old[r["old"]].append(r)
terms_by = m.terms()[1]
retired = {o for o in by_old if m.sites_of(o, terms_by, "x.md")}
print("old words", len(by_old), "retired by a row", len(retired), file=sys.stderr)
HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
def norm(t): return HUMP.sub("_", t).lower()
PART = re.compile(r"[a-z0-9]+")
def parts(s): return PART.findall(norm(s))
files = subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.decode().split("\0")
texts, ptoks = {}, {}
for f in files:
    if not f or f.startswith("tools/move/") or f == "GLOSSARY.md": continue
    p = Path(f)
    try:
        if not p.is_file() or p.stat().st_size > 1_000_000: continue
        t = p.read_text(encoding="utf-8")
    except Exception: continue
    texts[f] = t
nfiles = list(texts)
nnorm = {f: norm(texts[f]) for f in nfiles}
index = collections.defaultdict(set)
need = set(x for o in by_old if o not in retired for x in parts(o))
for i, f in enumerate(nfiles):
    for tok in set(PART.findall(nnorm[f])):
        if tok in need: index[tok].add(i)
print("indexed", len(nfiles), "files", file=sys.stderr)
res = []
for o in sorted(by_old):
    if o in retired: continue
    ps = parts(o)
    if not ps: continue
    cand = set.intersection(*(index.get(p, set()) for p in ps))
    if not cand: continue
    whole = re.compile(r"(?<!\w)" + re.escape(o) + r"(?!\w)", re.I)
    partre = re.compile(r"(?<![a-z0-9])" + r"[ _\-]?".join(map(re.escape, ps)) + r"(?![a-z0-9])")
    nw = np = 0; fs = set(); own = collections.Counter(); ex = []
    for i in cand:
        f = nfiles[i]; t = texts[f]; n = nnorm[f]
        w = len(whole.findall(t)); a = len(partre.findall(n))
        if not a and not w: continue
        pt = max(a - w, 0)
        nw += w; np += pt; fs.add(f); own[owner(f)] += w + pt
        if len(ex) < 3:
            for mo in (whole.search(t) or partre.search(n),):
                ex.append(f"{f}:{(t if whole.search(t) else n).count(chr(10), 0, mo.start()) + 1}")
    if not fs: continue
    inlint = sum(1 for f in fs if m.holds(f))
    res.append(dict(old=o, whole=nw, part=np, total=nw + np, files=len(fs), lint_files=inlint,
                    owners=", ".join(f"{k}:{v}" for k, v in own.most_common(6)), ex=" ".join(ex)))
res.sort(key=lambda r: -r["total"])
with open(out / "old_words.tsv", "w") as fh:
    cols = ["old", "new", "map", "commit", "scope", "whole_w", "part_of_identifier", "sites", "files", "files_the_lint_reads", "owner_lanes", "examples", "other_live_sense"]
    fh.write("\t".join(cols) + "\n")
    for r in res:
        ps = by_old[r["old"]]
        news = " | ".join(dict.fromkeys(p["new"] for p in ps))
        maps = " | ".join(dict.fromkeys(p["map"] for p in ps)); cm = " | ".join(dict.fromkeys(p["commit"] for p in ps))
        sc = " | ".join(dict.fromkeys(p["scope"] for p in ps))
        fh.write("\t".join(map(str, [r["old"], news, maps, cm, sc, r["whole"], r["part"], r["total"], r["files"], r["lint_files"], r["owners"], r["ex"], ""])) + "\n")
with open(out / "retired_by_row.txt", "w") as fh:
    fh.write("\n".join(sorted(retired)) + "\n")
print(len(res), "old words with a site now; sites", sum(r["total"] for r in res), file=sys.stderr)
