#!/usr/bin/env python3
"""The resume's last step: by-lead/<lane>.md and the reader checks over the tables.

  python3 finalize.py V2_DIR N2_DIR OVER_TSV OUT_DIR [GLOSSARY_REF]

V2_DIR holds old_words.v2.tsv, maps_not_run.tsv, both_live_sites.tsv (with its `reading` column filled by read_sites.py),
readings.tsv and readings_by_lane.tsv; N2_DIR holds names.v2.tsv and sites_attesters.tsv (names_v2.py).  Writes OUT_DIR/by-lead/
and OUT_DIR/reader_checks.txt (the reader's findings: a zero-site row with a class, a `new` spelling that is an older name of
GLOSSARY_REF's GLOSSARY.md, a HELD row that names no file)."""
from __future__ import annotations

import collections
import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
v2, n2, over_path, out = (Path(a) for a in sys.argv[1:5])
ref = sys.argv[5] if len(sys.argv) > 5 else "c41a7b5a6d5c59bb95363a01770633933a83c7c5"
csv.field_size_limit(1 << 28)


def rd(p, **kw):
    return list(csv.DictReader(open(p), delimiter="\t", **kw))


glossary = subprocess.run(["git", "show", f"{ref}:GLOSSARY.md"], capture_output=True, text=True).stdout
older: set[str] = set()
for line in glossary.split("\n"):
    if line.startswith("|") and not line.startswith("|---"):
        first = line.split("|")[1]
        older |= {a or b for a, b in re.findall(r'"([^"]+)"|`([^`]+)`', first)}
older_l = {o.lower() for o in older}

words = rd(v2 / "old_words.v2.tsv")
sites_all = rd(v2 / "both_live_sites.tsv")
by_lane_read = rd(v2 / "readings_by_lane.tsv")
names = rd(n2 / "names.v2.tsv")
att = rd(n2 / "sites_attesters.tsv")
over = rd(over_path, quoting=csv.QUOTE_NONE)

problems: list[str] = []
for r in names:
    if int(r["sites"]) == 0 and r["class"]:
        problems.append(f"names.v2: zero-site row with a class: {r['spelling']} / {r['context']}")
    for alt in re.split(r"\s*\|\s*", r["new"]):
        if alt and alt.lower() in older_l:
            problems.append(f"names.v2: new spelling {alt!r} (of {r['spelling']!r}) is an older name in GLOSSARY.md")
    if r["class"] == "HELD" and not re.search(r"[\w.-]+(?:/[\w.-]+)*\.(?:py|toml|json|md|sh|lean|lock|rs|jsonl|yml|yaml|txt)\b|\.service|moves\.json", r["evidence"]):
        problems.append(f"names.v2: HELD row names no file: {r['spelling']} / {r['context']}: {r['evidence'][:80]}")
for r in words:
    if int(r["old_sites"]) == 0 and r["verdict"] != "GONE":
        problems.append(f"old_words.v2: zero-site row not GONE: {r['old']}")
    if int(r["old_sites"]) > 0 and r["verdict"] == "GONE":
        problems.append(f"old_words.v2: GONE row with sites: {r['old']}")
    for alt in re.split(r"\s*\|\s*", r["new"]):
        if alt and alt.lower() in older_l:
            problems.append(f"old_words.v2: new spelling {alt!r} (of {r['old']!r}) is an older name in GLOSSARY.md")
if not any(r["class"] for r in names):
    problems.append("names.v2: no class at all")
(out).mkdir(parents=True, exist_ok=True)
(out / "reader_checks.txt").write_text(
    f"reader over names.v2.tsv ({len(names)} rows), old_words.v2.tsv ({len(words)} rows); GLOSSARY.md at {ref[:12]} names {len(older)} older spellings\n"
    + ("\n".join(problems) if problems else "no finding: no zero-site row carries a class, no new spelling is an older name, every HELD row names a file") + "\n")
print(len(problems), "reader findings")

new_of = {(r["term"], r["spelling"], r["context"]): r["new"] for r in names}
read_words = {r["old"] for r in by_lane_read}
new_word = {r["old"]: r["new"] for r in words}
listed_words_by_lane = collections.defaultdict(set)
for s in sites_all:
    if s["old"] not in read_words:
        listed_words_by_lane[s["owner"]].add(s["old"])

sun = [r for r in over if "2026-11-15T08:00Z" <= r["commit_date"] < "2026-11-16T08:00Z" and r["added_after_entry"] != "no"]
lanes = sorted({r["owner"] for r in sun} | {r["lane"] for r in by_lane_read} | {r["owner"] for r in att if r["class"] in ("RENAME", "ASK")})
(out / "by-lead").mkdir(exist_ok=True)


def fl(f, l):
    return f"`{f}:{l}`"


for lane in lanes:
    fname = lane.strip("()") if lane != "(none)" else "unowned"
    lines = [f"# {lane}: what the sweep found in your files (base c41a7b5a6d5c)", ""]
    mine = [r for r in sun if r["owner"] == lane]
    lines += ["## (a) Old-word sites your PRs added past their file's entry on Sun 15 Nov (Pacific day, 08:00Z to 08:00Z)", ""]
    if mine:
        by_pr = collections.defaultdict(list)
        for r in mine:
            by_pr[r["pr"]].append(r)
        lines += ["| PR | sites | words | examples |", "|---|---:|---|---|"]
        for pr, rs in sorted(by_pr.items(), key=lambda kv: -len(kv[1])):
            ws = collections.Counter(r["old"] for r in rs).most_common(4)
            lines.append(f"| {pr} | {len(rs)} | {', '.join(f'{w} {n}' for w, n in ws)} | {', '.join(fl(r['file'], r['line']) for r in rs[:3])} |")
        lines += ["", "A site counts when its file holds more sites than the lint's entry allows and the line was added after the entry was set, or the file has no entry. `over_entries.tsv` has every line."]
    else:
        lines.append("None.")
    lines += ["", "## (b) Your BOTH LIVE words (old and new spelling both have sites), counts as read", ""]
    rows = [r for r in by_lane_read if r["lane"] == lane]
    if rows:
        lines += ["| old | new | your sites | opened in your files | held among those opened |", "|---|---|---:|---:|---:|"]
        for r in sorted(rows, key=lambda r: -int(r["sites_in_lane"])):
            lines.append(f"| `{r['old']}` | {new_word.get(r['old'], '')[:60].replace('|', '/')} | {r['sites_in_lane']} | {r['opened_in_lane']} | {r['held_among_opened']} |")
        lines += ["", "Each row's old sites are in `both_live_sites.tsv` with the `reading` column filled for the opened ones."]
    else:
        lines.append("No BOTH LIVE word with over 50 old sites has a site in your files.")
    lw = sorted(listed_words_by_lane.get(lane, ()))
    if lw:
        lines += ["", f"{len(lw)} more BOTH LIVE words, each with 50 or fewer old sites in the whole tree, have sites in your files. I listed them and opened none, so I give no count: filter `both_live_sites.tsv` by owner `{lane}`."]
    lines += ["", "## (c) Your RENAME lines among today's words", ""]
    ren = collections.defaultdict(list)
    for r in att:
        if r["owner"] == lane and r["class"] == "RENAME":
            ren[(r["term"], r["context"])].append(r)
    if ren:
        lines += ["| word | context | your sites | option | paths (sites) |", "|---|---|---:|---|---|"]
        for (term, ctx), rs in sorted(ren.items(), key=lambda kv: -len(kv[1])):
            opts = collections.Counter((r["spelling"], new_of.get((term, r["spelling"], ctx), "")) for r in rs).most_common(3)
            opt = "; ".join(f"`{a}` -> `{b}` ({n})" for (a, b), n in opts)
            paths = ", ".join(f"`{f}` {n}" for f, n in collections.Counter(r["file"] for r in rs).most_common(3))
            lines.append(f"| {term} | {ctx[:50]} | {len(rs)} | {opt} | {paths} |")
    else:
        lines.append("None.")
    lines += ["", "## (d) Your ASK lines", ""]
    ask = collections.defaultdict(list)
    for r in att:
        if r["owner"] == lane and r["class"] == "ASK":
            ask[r["term"]].append(r)
    asked = False
    for term, rs in ask.items():
        ex = ", ".join(fl(r["file"], r["file"] and r["line"]) for r in rs[:3])
        if term == "certifier":
            lines.append(f"- {len(rs)} bare `certifier` sites in your files (for example {ex}): is the word the device (it becomes `network attester`) or the network protocol (it stays)? Answer `device` or `protocol`.")
        else:
            lines.append(f"- {len(rs)} bare `the challenger` sites on lines about memory (for example {ex}): is it the memory attester or the audit's challenger? Answer `attester` or `challenger`.")
        asked = True
    for term, word, answer in (("unit", "unit", "`proof block` or `other`"), ("row/cell", "row", "`result` or `other`")):
        if any(r["term"] == term and r["class"] == "ASK" and re.search(rf"(?:^|, ){re.escape(lane)}:", r["owners"]) for r in names):
            sense = "a proof block" if term == "unit" else "a benchmark result"
            lines.append(f"- Bare `{word}` and `{word}s` in your files carry no cue on many lines (rows `{term}` of `names.v2.tsv` list the lanes; I give no count, the class came from a rule and not from reading each site): does it mean {sense} where your files use it? Answer {answer}.")
            asked = True
    if not asked:
        lines.append("None.")
    (out / "by-lead" / f"{fname}.md").write_text("\n".join(lines) + "\n")
print(len(lanes), "lanes:", ", ".join(lanes))
