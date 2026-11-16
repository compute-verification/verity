#!/usr/bin/env python3
"""Final tables of parts 2 to 4 from the runs' raw output.
Usage: python3 assemble.py TODAY_DIR(t2) PATHS_DIR(t3) OVER_TSV GATE_DIR OUT_DIR
 names.tsv (part 2): names.tsv's columns plus class_if_renamed; the certifier and memory challenger lines are ASK (no
 Glossary row names a replacement), the rest as classed by today_sites.py.
 path_names.tsv (3a): row/cell only under benchmarks/ and tools/research/ (elsewhere a row is a matrix's or a table's).
 over_entries.tsv (4): the sites the lint's entries do not hold (no entry for the file, or the line came after the entry).
 gate_verdicts.tsv (4): the ratchet gate's verdict on each merge that added sites."""
import csv, re, shutil, sys
from pathlib import Path

t2, t3, over, gate, out = (Path(a) for a in sys.argv[1:6]); out.mkdir(parents=True, exist_ok=True)

def rd(p): return list(csv.DictReader(open(p), delimiter="\t"))
def wr(p, rows, cols):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(rows)

rows = rd(t2 / "names2.tsv")
for r in rows:
    r["class_if_renamed"] = r["class"]
    if r["term"] in ("certifier", "memory challenger"):
        r["new"] = "(open: no Glossary row names the attesters' word)"
        r["evidence"] = ("one question: which word replaces it for the two attesters (the network certifier, the memory "
                         "challenger)? if renamed it would be " + r["class"] + (": " + r["evidence"] if r["evidence"] else ""))
        r["class"] = "ASK"
    else:
        if r["new"].endswith("(auto)"):
            r["new"] = r["new"][:-7]
        sp = r["spelling"]
        if " " in r["new"] and ("-" in sp or "_" in sp):
            r["new"] = r["new"].replace(" ", "-" if "-" in sp else "_")
        bare = re.fullmatch(r"(?i)(unit|units|row|rows|cell|cells)", sp)
        if bare:
            r["new"] = {"unit": "proof block", "row": "result", "cell": "result"}[re.sub("s$", "", sp.lower())] + ("s" if sp.lower().endswith("s") else "")
            if sp.isupper() or sp[0].isupper():
                r["new"] = r["new"].upper() if sp.isupper() else r["new"].capitalize()
            r["evidence"] = ("one question: is this site the " + ("proof block" if "unit" in sp.lower() else "benchmark result")
                             + " sense? the bare word is also a unit of measure, a unit test, a matrix or table row, a commitment cell; "
                             "the owner reads each site (sites.tsv)")
            r["class"] = "ASK"
        elif sp in ("wrong_units", "proved_units"):
            r["class"] = "HELD"
            r["evidence"] = "GLOSSARY.md Older names: unit in the audit record's keys `wrong_units` and `proved_units`, until a format version"
cols = ["spelling", "term", "sites", "files", "lean_sites", "owners", "examples", "new", "kind", "class", "evidence", "class_if_renamed"]
wr(out / "names.tsv", rows, cols)
for n in ("sites2.tsv", "phrases.tsv", "by_owner2.tsv", "row_cell_files.tsv", "excluded.tsv"):
    shutil.copy(t2 / n, out / n.replace("2.tsv", ".tsv"))

pn = [r for r in rd(t3 / "path_names.tsv")
      if not (r["term"] == "today:row/cell" and not r["path"].startswith(("benchmarks/", "tools/research/")))]
wr(out / "path_names.tsv", pn, ["path", "kind", "term", "files_under", "owner"])
shutil.copy(t3 / "skipped_areas.tsv", out / "skipped_areas.tsv"); shutil.copy(t3 / "skipped_areas_summary.tsv", out / "skipped_areas_summary.tsv")
cm = rd(t3 / "commands.tsv")
for r in cm:
    if r["old"] in ("cap", "caps") and r["file"].startswith(("tools/verity/lean/", "tools/check/")):
        r["class"] = "OTHER SENSE"
        r["evidence"] = "the Lean check's caps file and counts (a ceiling on a count), not Pearl's cap: the lint's OTHER_SENSES should gain it"
    elif r["old"] in ("certifier", "memory challenger"):
        r["class"] = "ASK"; r["evidence"] = "one question: which word replaces it for the two attesters?"
wr(out / "commands.tsv", cm, list(cm[0]))

ov = rd(over)
past = [r for r in ov if r["added_after_entry"] in ("yes", "no entry")]
wr(out / "over_entries.tsv", past, ["file", "line", "old", "commit", "pr", "commit_date", "commit_subject", "found_allowed", "owner",
                                    "entry_last_set", "added_after_entry"])
gr = []
for l in open(gate / "summary.txt"):
    m = re.match(r"#(\d+) merge (\w+) exit (\d+)", l)
    if m:
        txt = (gate / f"{m.group(1)}.txt").read_text()
        v = re.search(r"failed: (adds \d+ [a-z-]+ site\(s\)(?: to the files it changes)?.*?\));", txt, re.S) or re.search(r"failed: (.{0,200})", txt, re.S)
        gr.append({"pr": "#" + m.group(1), "merge": m.group(2), "exit": m.group(3), "verdict": "passed" if m.group(3) == "0" else "failed",
                   "first_finding": (v.group(1) if v else "")[:220].replace("\n", " ")})
wr(out / "gate_verdicts.tsv", gr, ["pr", "merge", "exit", "verdict", "first_finding"])
print(len(rows), "names;", len(pn), "path names;", len(past), "of", len(ov), "over-entry sites;", len(gr), "gate verdicts")
