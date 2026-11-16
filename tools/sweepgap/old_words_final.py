#!/usr/bin/env python3
"""Part 1's table from old_words.tsv (map_words_sites.py): adds, per old word, whether a Glossary "Older names" row
names it (the lint's matcher reads only plain words; a row for a sense-bound word like "window" is named but not read),
and its other live sense: the map's own [keep] patterns that contain the word, or its scope (a word a map renamed only in
listed files is another word elsewhere). Splits the table into the words no row names (old_words_final.tsv) and the words
a row names that the lint cannot read (listed_not_read.tsv).
Run from the checkout root: python3 old_words_final.py OLD_WORDS.tsv OUT_DIR"""
import csv, re, subprocess, sys, tomllib
from pathlib import Path

src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(src), delimiter="\t"))

older = []
on = False
for line in Path("GLOSSARY.md").read_text(encoding="utf-8").split("\n"):
    if line.startswith("## "):
        on = line.strip() == "## Older names"
    elif on and line.startswith("|") and not line.startswith("| ---") and not line.startswith("| Old"):
        cells = line.strip("|").split(" | ")
        older.append(cells[0].lower())

TOKENS = {}
for cell in older:
    for q1, q2 in re.findall(r'"([^"]+)"|`([^`]+)`', cell):
        TOKENS.setdefault((q1 or q2).strip(), cell)

def names_it(old):
    o = old.lower()
    for tok, cell in TOKENS.items():
        if tok == o or tok.rstrip("*") == o.rstrip("*") or (tok.endswith("*") and o.startswith(tok.rstrip("*"))):
            return tok
    return ""

HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
SINGLE = {k.lower() for k in TOKENS if re.fullmatch(r"[A-Za-z]+", k)}

def part_named(old):
    if not re.fullmatch(r"[A-Za-z0-9_ -]+", old):
        return ""
    ps = re.findall(r"[a-z0-9]+", HUMP.sub("_", old).lower())
    return ",".join(p for p in ps if p in SINGLE)

def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout

_keeps = {}
def keeps_of(mapname, commit):
    key = (mapname, commit)
    if key in _keeps:
        return _keeps[key]
    txt = git("show", f"{commit}:tools/move/{mapname}") if commit else ""
    if not txt:
        p = Path("tools/move") / mapname
        txt = p.read_text() if p.exists() else ""
    pats = []
    for m in re.finditer(r"(?ms)^\[keep\]\s*(.*?)(?=^\[|\Z)", txt):
        pats += re.findall(r"'([^'\n]+)'|\"([^\"\n]+)\"", m.group(1))
    pats = [a or b for a, b in pats]
    _keeps[key] = pats
    return pats

def keep_note(r):
    olds = r["old"].lower()
    maps = [x.strip() for x in r["map"].split("|")]
    commits = [x.strip() for x in r["commit"].split("|")]
    hits = []
    for i, mp in enumerate(maps):
        c = commits[i] if i < len(commits) else commits[-1]
        for p in keeps_of(mp, c):
            if olds in re.sub(r"[\\\[\]()?*+^$]|\(\?[^)]*\)", "", p).lower():
                hits.append(f"{mp}: {p[:60]}")
                break
    return hits

JUDGED = {  # read by me: a sense the map keeps, or ordinary English
    "registration": "yes: the registration log, the `registration` protocol and the registered values keep the word (AGENTS.md); the map renamed it only where it meant the docket",
    "registrations": "yes: same as registration (plural)",
    "registration record": "yes: same as registration; the registration log is the live phrase",
    "drew": "yes: ordinary English past tense, and a person's name in a few files; the map kept those",
    "undrawn": "yes: ordinary English; the map kept a [keep] pattern for it",
    "redrawn": "yes: ordinary English (figures, plots) beside the sampling sense",
    "redraws": "yes: ordinary English (figures, plots) beside the sampling sense",
    "novel": "yes: ordinary English adjective (a novel result); the Glossary row for NCI spells its own phrase",
    "epoch": "yes: a training epoch (torch), and the time base of a timestamp, beside the execution phase sense",
    "auditor host": "yes: the host of an auditor's device is the `trusted-device` assumption's own phrase (AGENTS.md: the device and its host)",
    "sample": "yes: the Glossary's own word (a sample of proof blocks); the map renamed only the budget's plan point",
    "remote_session": "no: `remote` stays for the remote auditor (`compose/verifier/remote/`); the compound named the PoUB session",
}

fin, listed = [], []
for r in rows:
    r["part_named"] = part_named(r["old"]) if len(r["old"]) else ""
    r["glossary_names_it"] = names_it(r["old"]) or "no"
    hits = keep_note(r)
    scoped = r["applies"].startswith("only in")
    why = []
    if scoped:
        why.append("yes: the map renames it only in " + r["applies"][8:80] + "; elsewhere it is another word (sites_tree_wide)")
    if hits:
        why.append("yes: the map's [keep] holds it: " + "; ".join(hits[:2]))
    if r["old"].lower() in JUDGED:
        why.insert(0, JUDGED[r["old"].lower()])
    r["other_live_sense"] = " | ".join(why) if why else "no sign (not scoped, no [keep] pattern holds it)"
    (listed if r["glossary_names_it"] != "no" else fin).append(r)

cols = ["old", "new", "map", "commit", "applies", "sites", "whole_w", "part_of_identifier", "sites_tree_wide", "files",
        "files_the_lint_reads", "owner_lanes", "examples", "glossary_names_it", "part_named", "other_live_sense"]
for name, rs in (("old_words_final.tsv", fin), ("listed_not_read.tsv", listed)):
    rs.sort(key=lambda x: -int(x["sites"]))
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(rs)
print("no row names it:", len(fin), "a row names it, the lint can't read it:", len(listed))
