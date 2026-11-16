#!/usr/bin/env python3
"""Part 2 against the landed Glossary.  Run from the checkout root after today_sites.py:
    python3 names_v2.py T2_DIR OUT_DIR
certifier: GLOSSARY.md's row retires "network certifier" and "the certifier" for the device (the network attester).  Per
(spelling, context): LEAN for a Lean site; HELD where the Glossary row, a recorded file, a deploy file or a caller outside
the tree reads the spelling (evidence: the file and line); OTHER SENSE where the bare word names the network protocol
(its id, area, audit, claim ids), not the device; RENAME where the row's word applies and nothing holds it; ASK where
a bare "the certifier" shows no cue.
memory challenger: "memory attester" in prose only tonight: prose sites RENAME, identifiers, keys and strings NAME-LATER; a bare
"the challenger" on a line about memory is ASK.
unit, row/cell, proof table: as the first pass left them."""
from __future__ import annotations

import collections
import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan as S

t2, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
m = S.lint(); owner = S.owners()
GLOSSARY_ROW = next((i for i, l in enumerate(Path("GLOSSARY.md").read_text(encoding="utf-8").split("\n"), 1)
                     if l.startswith('| "network certifier"')), 0)
files = [f for f in subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.decode().split("\0") if f]
cache: dict[str, list[str]] = {}


def lines_of(f: str) -> list[str]:
    if f not in cache:
        try:
            cache[f] = Path(f).read_text(encoding="utf-8").split("\n")
        except Exception:
            cache[f] = []
    return cache[f]


HELD_FILE = re.compile(r"(^|/)(fixtures|records)/|moves\.json$|lean-audit\.json$|\.lock$|TRUSTED\.sha256$|^tools/research/src/research/store/"
                       r"|^infra/|\.service$|\.timer$|(^|/)deploy\.toml$|^artifacts\.json$|^tests/lint/allowlists/")
CERT_TOK = re.compile(r"[\w./:@-]*certifier[\w./:@-]*", re.I)
CHAL_TOK = re.compile(r"[\w./:@-]*memory[-_ ]?challenger[\w./:@-]*", re.I)
held_tok: dict[str, str] = {}
shell_tok: dict[str, str] = {}
for f in files:
    if f.startswith(("tools/move/", "tools/sweepgap/")) or f == "GLOSSARY.md":
        continue
    held = bool(HELD_FILE.search(f)) or any(d in m.FIXTURE_DIRS for d in f.split("/")[:-1])
    shell = f.endswith((".sh", ".bash", ".toml", ".yaml", ".yml", ".service", ".timer", ".conf")) and not f.endswith("metrics.toml")
    metrics = f.endswith("metrics.toml")
    if not (held or shell or metrics):
        continue
    for i, ln in enumerate(lines_of(f), 1):
        if "ertifier" not in ln and "hallenger" not in ln:
            continue
        if metrics and not ln.lstrip().startswith("id"):
            continue
        for mo in CERT_TOK.finditer(ln):
            for piece in re.split(r"[./:@]", mo.group(0)):
                (held_tok if (held or metrics) else shell_tok).setdefault(piece.lower(), f"{f}:{i}")

GLOSSARY_HELD = ("verity-network-certifier", "network-certifier", "network_certifier_run")
PROTO_CUE = re.compile(r"""(?ix)["'`]certifier["'`]|certifier[-_/ ](?:coverage|protocol|area|audit|check|program|package|verifier|spec)
    |(?:protocol|area|package|subprotocol)[ `'"]*certifier|certifier['’]?s?[ ]+(?:protocol|area|audit|check|guarantee|theorem|proof|lean|PROTOCOL|
    verifier|program|package|team|lane|spec|claim)|/certifier/|certifier/|certifier\s*=|\bcertifier\s*:\s""")
DEVICE_CUE = re.compile(r"""(?ix)certifier['’]?s?[ ]+(?:sign|signs|signed|key|keys|device|clock|stamp|stamps|release|releases|attach|attachment|link|
    tick|bucket|cell|cells|witness|sees|saw|reads|delivers|order|ingress|egress|wipe|host|front|serves|record|records|reveal|salt|piece|
    session|sessions|answer|answers|count|certificate|certificates|period|window|windows|response|request|offline|honest|trusted)
    |(?:a|each|no|its|whose|own|honest|another|second|active|split|older|that|any|every|one)[ ]+certifier\b|(?:ingress|in-process|tick|fifo|network)[ -]certifier""")
NET_PHRASE = re.compile(r"(?i)network[ _-]certifiers?")


def spell_of(sp: str, matched: str) -> str:
    return sp if len(sp) > len(matched) else matched


rows = list(csv.DictReader(open(t2 / "sites2.tsv"), delimiter="\t"))
ctx_rows = collections.OrderedDict()
site_out = []


def add(term, spelling, context, f, ln, kind, own, cls, new, evidence):
    k = (term, spelling, context, cls, new)
    d = ctx_rows.setdefault(k, {"sites": 0, "files": set(), "lean": 0, "own": collections.Counter(), "ex": [], "kinds": collections.Counter(), "ev": evidence})
    d["sites"] += 1; d["files"].add(f); d["lean"] += f.endswith(".lean"); d["own"][own] += 1; d["kinds"][kind] += 1
    if len(d["ex"]) < 3:
        d["ex"].append(f"{f}:{ln}")
    site_out.append((term, f, ln, spelling, kind, own, context, cls))


def attester_of(sp: str, bare_prose: bool) -> str:
    s = sp
    if re.fullmatch(r"(?i)certifiers?", s):
        pl = s.lower().endswith("s")
        return ("network attesters" if pl else "network attester") if bare_prose else ("attesters" if pl else "attester")
    s = re.sub(r"certifier", "attester", s); s = re.sub(r"Certifier", "Attester", s); s = re.sub(r"CERTIFIER", "ATTESTER", s)
    return s


for r in rows:
    term = r["term"]
    if term not in ("certifier", "memory challenger"):
        continue
    f, ln, kind, own = r["file"], int(r["line"]), r["kind"], r["owner"]
    line = lines_of(f)[ln - 1] if ln - 1 < len(lines_of(f)) else ""
    sp = spell_of(r["spelling"], r["matched"])
    if term == "memory challenger":
        if f.endswith(".lean") and not re.fullmatch(r"(?i)memory challengers?", sp):
            add(term, sp, "Lean name", f, ln, kind, own, "NAME-LATER", "", "a Lean name: lean's names pass, in a later PR")
        elif kind == "prose" or f.endswith(".lean"):
            add(term, sp, "prose", f, ln, kind, own, "RENAME", re.sub(r"(?i)challenger", lambda mo: "Attester" if mo.group(0)[0] == "C" else "attester", sp), "")
        else:
            add(term, sp, kind, f, ln, kind, own, "NAME-LATER", "", "an identifier, key or string: the Glossary row and the names move come in a later PR; memory-accounting renames its own files")
        continue
    low = sp.lower()
    if f.endswith(".lean") and re.fullmatch(r"(?i)certifiers?", sp) and re.search(r"(?i)network certifiers?", line):
        add(term, sp, '"network certifier" in a Lean line', f, ln, kind, own, "RENAME", "network attester" if low == "certifier" else "network attesters",
            "the words map renames it in a comment or docstring; a string literal stays")
        continue
    if f.endswith(".lean"):
        add(term, sp, "Lean", f, ln, kind, own, "LEAN", attester_of(sp, False),
            f"GLOSSARY.md:{GLOSSARY_ROW}: the Lean package `NetworkCertifier` and the names that follow it stay until lean renames them")
        continue
    if any(low == g or low.startswith(g + "/") or low.startswith(g + "-") for g in GLOSSARY_HELD) or low in GLOSSARY_HELD \
            or re.fullmatch(r"verity-network-certifier(?:/.*)?", low) or '"network_certifier"' in line:
        add(term, sp, "held by the Glossary row", f, ln, kind, own, "HELD", "", f"GLOSSARY.md:{GLOSSARY_ROW}: its domain tags, the party `network-certifier`, the record keys `\"network_certifier\"` and `network_certifier_run`")
        continue
    bare = low in ("certifier", "certifiers")
    if HELD_FILE.search(f) or any(d in m.FIXTURE_DIRS for d in f.split("/")[:-1]):
        add(term, sp, "in a recorded or deployed file", f, ln, kind, own, "HELD", "", f"{f}:{ln} is a recorded, locked or deployed file that holds the spelling")
        continue
    if not bare and low in held_tok:
        add(term, sp, "held by a recorded or deployed file", f, ln, kind, own, "HELD", "", f"{held_tok[low]} holds the spelling")
        continue
    if not bare and low in shell_tok:
        add(term, sp, "read by a shell, deploy or configuration file", f, ln, kind, own, "HELD", "", f"{shell_tok[low]} reads the spelling")
        continue
    if low in ("certifier", "certifiers"):
        if NET_PHRASE.search(line):
            add(term, sp, '"network certifier" (the row\'s phrase)', f, ln, kind, own, "RENAME", "network attester" if low == "certifier" else "network attesters", "")
        elif PROTO_CUE.search(line):
            add(term, sp, "the protocol (its id, area, audit or claim ids)", f, ln, kind, own, "OTHER SENSE", "",
                "the bare word names the network protocol, not the device: no row renames it (tools/move/ontology_words_map.toml's note)")
        elif DEVICE_CUE.search(line):
            add(term, sp, '"the certifier" for the device (the row)', f, ln, kind, own, "RENAME", "the network attester" if low == "certifier" else "the network attesters", "")
        else:
            add(term, sp, "bare, no cue on the line", f, ln, kind, own, "ASK", "",
                f"is this the device (then the network attester) or the protocol (then the word stays)?")
        continue
    add(term, sp, "identifier", f, ln, kind, own, "RENAME", attester_of(sp, False), "")

# bare "the challenger" on a line about memory
CHAL_CUE = re.compile(r"(?i)memory|wipe|poub|pous|sealed|\bfill|\bhbm|\bpage|store|gib")
THE_CH = re.compile(r"(?i)\bthe challengers?\b")
for f in files:
    if f.startswith(("tools/move/", "tools/sweepgap/")) or f == "GLOSSARY.md" or not f.endswith((".py", ".md", ".sh", ".toml", ".txt", ".rst", ".rs", ".cu")):
        continue
    if f.endswith(".json") or any(d in m.FIXTURE_DIRS for d in f.split("/")[:-1]):
        continue
    for i, ln in enumerate(lines_of(f), 1):
        if "hallenger" in ln and THE_CH.search(ln) and CHAL_CUE.search(ln) and not re.search(r"(?i)memory[ _-]?(challenger|attester)", ln):
            add("memory challenger", "the challenger", "bare, on a line about memory", f, i, "prose", owner(f), "ASK", "",
                "does this name the memory attester, or the challenger of the audit (the auditor's party)? answer with the word")

with open(out / "names.v2.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["spelling", "term", "context", "sites", "files", "lean_sites", "owners", "examples", "new", "kind", "class", "evidence"])
    for (term, sp, ctx, cls, new), d in sorted(ctx_rows.items(), key=lambda kv: -kv[1]["sites"]):
        ev = d["ev"]
        w.writerow([sp, term, ctx, d["sites"], len(d["files"]), d["lean"], ", ".join(f"{o}:{n}" for o, n in d["own"].most_common(6)),
                    " ".join(d["ex"]), new, d["kinds"].most_common(1)[0][0], cls, ev])
with open(out / "sites_attesters.tsv", "w") as fh:
    fh.write("term\tfile\tline\tspelling\tkind\towner\tcontext\tclass\n")
    for r in site_out:
        fh.write("\t".join(map(str, r)) + "\n")
print(collections.Counter(r[7] for r in site_out))


# unit, row/cell, proof table: names2.tsv's rule-set classes, with the first pass's adjustments (assemble.py)
legacy = [r for r in csv.DictReader(open(t2 / "names2.tsv"), delimiter="\t") if r["term"] in ("unit", "row/cell", "proof table")]
with open(out / "names.v2.tsv", "a", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    for r in legacy:
        sp = r["spelling"]
        new = r["new"][:-7] if r["new"].endswith("(auto)") else r["new"]
        cls, ev = r["class"], r["evidence"]
        if " " in new and ("-" in sp or "_" in sp):
            new = new.replace(" ", "-" if "-" in sp else "_")
        if re.fullmatch(r"(?i)(unit|units|row|rows|cell|cells)", sp):
            base = {"unit": "proof block", "row": "result", "cell": "result"}[re.sub("s$", "", sp.lower())] + ("s" if sp.lower().endswith("s") else "")
            new = base.upper() if sp.isupper() else (base.capitalize() if sp[0].isupper() else base)
            ev = ("one question: is this site the " + ("proof block" if "unit" in sp.lower() else "benchmark result") + " sense? the bare word is also a unit of "
                  "measure, a unit test, a matrix or table row, a commitment cell; the owner reads each site (sites.v2.tsv.gz)")
            cls = "ASK"
        elif sp in ("wrong_units", "proved_units"):
            cls, ev = "HELD", "GLOSSARY.md Older names: unit in the audit record's keys `wrong_units` and `proved_units`, until a format version"
        w.writerow([sp, r["term"], "", r["sites"], r["files"], r["lean_sites"], r["owners"], r["examples"], new, r["kind"], cls, ev])
