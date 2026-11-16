#!/usr/bin/env python3
"""For each old word of pairs.tsv (maps_words.py): is it retired by a row of GLOSSARY.md (the lint's matcher)? If not, its
sites in today's tree (tracked text files under 1 MB, not tools/move/ nor GLOSSARY.md): whole (`git grep -n -i -w`: a match
no word character touches), and as one part of an identifier (snake, camel, capitals), by owner lane, with three examples.
A word a map renamed only in listed files ([[words_in]]) is counted in those files that still stand at their path
(`sites_in_scope`), and tree-wide beside it (`sites_tree`): a scoped word elsewhere is another sense of it.
Run from the checkout root:  python3 map_words_sites.py PAIRS.tsv OUT_DIR"""
import collections, csv, importlib.util, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan
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
PLAIN = re.compile(r"^[A-Za-z0-9_ -]+$")
def parts(s): return PART.findall(norm(s)) if PLAIN.match(s) else []
files = subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.decode().split("\0")
texts = {}
for f in files:
    if not f or f.startswith("tools/move/") or f == "GLOSSARY.md": continue
    p = Path(f)
    try:
        if not p.is_file() or p.stat().st_size > 1_000_000: continue
        texts[f] = p.read_text(encoding="utf-8")
    except Exception: continue
nfiles = list(texts); idx_of = {f: i for i, f in enumerate(nfiles)}
nnorm = {f: norm(texts[f]) for f in nfiles}
index = collections.defaultdict(set)
lit_need = {}   # non-plain olds: a literal rare piece to find candidates by
need = set()
for o in by_old:
    if o in retired: continue
    ps = parts(o)
    if ps: need.update(ps)
    else:
        lit_need[o] = max(re.findall(r"[A-Za-z0-9_]+", o) or [""], key=len).lower()
        if lit_need[o]: need.add(lit_need[o])
for i, f in enumerate(nfiles):
    for tok in set(PART.findall(nnorm[f])):
        if tok in need: index[tok].add(i)
    for tok in set(re.findall(r"[a-z0-9_]+", texts[f].lower())):
        if tok in need and tok not in index: pass
print("indexed", len(nfiles), "files", file=sys.stderr)
def lit_rx(o):
    return re.compile((r"(?<![A-Za-z0-9_])" if re.match(r"\w", o[0]) else "") + re.escape(o) + (r"(?![A-Za-z0-9_])" if re.match(r"\w", o[-1]) else ""), re.I)
def count(o, cand):
    ps = parts(o)
    whole = lit_rx(o)
    partre = re.compile(r"(?<![a-z0-9])" + r"[ _\-]?".join(map(re.escape, ps)) + r"(?![a-z0-9])") if ps else None
    nw = np = 0; fs = set(); own = collections.Counter(); ex = []
    for i in sorted(cand):
        f = nfiles[i]; t = texts[f]
        w = len(whole.findall(t)); a = len(partre.findall(nnorm[f])) if partre else 0
        if not a and not w: continue
        pt = max(a - w, 0)
        nw += w; np += pt; fs.add(f); own[owner(f)] += w + pt
        if len(ex) < 3:
            mo = whole.search(t); ln = t.count("\n", 0, mo.start()) + 1 if mo else nnorm[f].count("\n", 0, partre.search(nnorm[f]).start()) + 1
            ex.append(f"{f}:{ln}")
    return nw, np, fs, own, ex
res = []
for o in sorted(by_old):
    if o in retired: continue
    ps = parts(o)
    key = ps if ps else ([lit_need[o]] if lit_need.get(o) else [])
    if not key: continue
    cand = set.intersection(*(index.get(p, set()) for p in key))
    if not cand: continue
    nw, np, fs, own, ex = count(o, cand)
    if not fs: continue
    scoped = [p for p in by_old[o] if p["scope"] == "words"]
    scope_files = sorted({f for p in by_old[o] if p["scope"].startswith("words_in:") for f in p["scope"][9:].split(";") if f})
    in_scope = {idx_of[f] for f in scope_files if f in idx_of}
    sc = count(o, cand & in_scope) if (not scoped and in_scope) else None
    res.append(dict(old=o, whole=nw, part=np, total=nw + np, files=len(fs), lint_files=sum(1 for f in fs if m.holds(f)),
                    owners=", ".join(f"{k}:{v}" for k, v in own.most_common(6)), ex=" ".join(ex), scoped_only=not scoped,
                    in_scope=(sc[0] + sc[1]) if sc else (None if scoped else 0), scope_ex=" ".join(sc[4]) if sc else "",
                    scope_owners=", ".join(f"{k}:{v}" for k, v in sc[3].most_common(6)) if sc else ""))
res.sort(key=lambda r: -(r["total"] if not r["scoped_only"] else (r["in_scope"] or 0)))
with open(out / "old_words.tsv", "w") as fh:
    cols = ["old", "new", "map", "commit", "applies", "sites", "whole_w", "part_of_identifier", "sites_tree_wide", "files", "files_the_lint_reads", "owner_lanes", "examples", "other_live_sense"]
    fh.write("\t".join(cols) + "\n")
    for r in res:
        ps = by_old[r["old"]]
        if r["scoped_only"] and not r["in_scope"]:
            continue
        news = " | ".join(dict.fromkeys(p["new"] for p in ps)); maps = " | ".join(dict.fromkeys(p["map"] for p in ps)); cm = " | ".join(dict.fromkeys(p["commit"] for p in ps))
        applies = "whole tree" if not r["scoped_only"] else "only in " + ("; ".join(dict.fromkeys(f for p in ps for f in p["scope"][9:].split(";")))[:120])
        sites = r["total"] if not r["scoped_only"] else r["in_scope"]
        fh.write("\t".join(map(str, [r["old"], news, maps, cm, applies, sites, r["whole"] if not r["scoped_only"] else "", r["part"] if not r["scoped_only"] else "", r["total"], r["files"], r["lint_files"],
                                     r["owners"] if not r["scoped_only"] else r["scope_owners"], r["ex"] if not r["scoped_only"] else r["scope_ex"], ""])) + "\n")
with open(out / "retired_by_row.txt", "w") as fh:
    fh.write("\n".join(sorted(retired)) + "\n")
print(len(res), "old words with a site now", file=sys.stderr)
