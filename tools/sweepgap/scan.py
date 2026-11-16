#!/usr/bin/env python3
"""Every site of a retired term in a checkout of verity, grouped by the spelling that stands in the tree.

Run from the repository root:  python3 scan.py [--out DIR] [--term REGEX] [--not-term REGEX] [--say REGEX] [--not-say REGEX]

It loads tests/lint/test_repo_retired_terms.py from the checkout (so GLOSSARY.md's table of older names is the term
source, with the lint's own rules for other senses and held keys), scans every tracked file the lint reads (about 25 s,
well under 1 GB), and writes into DIR (default /tmp/sweep):

  sites.tsv   one line a site: file, line, old name as the lint matched it, the Glossary's term, owner lane,
              the spelling (the whole identifier around the match, or the matched words when it stands in prose)
  names.tsv   one line a distinct (spelling, term): sites, files, .lean sites, owner lanes with counts, three examples,
              and four empty columns for the sweep to fill: new, kind, class, evidence
  by_owner.tsv  sites by owner lane and term

--term keeps only sites whose old name matches REGEX (case-insensitive); --not-term drops those that match.
--say and --not-say do the same on the Glossary's term (what the lint says to write instead), matched from its start.
Owner lane: the longest `owns` pattern of .agents/skills/using-slack/registry.json that the file falls under.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import importlib.util
import json
import re
import sys
from pathlib import Path

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.\-/][A-Za-z_][A-Za-z0-9_]*)*")


def lint():
    spec = importlib.util.spec_from_file_location("rt", "tests/lint/test_repo_retired_terms.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["rt"] = m
    spec.loader.exec_module(m)
    return m


def owners():
    reg = json.loads(Path(".agents/skills/using-slack/registry.json").read_text())
    pats = []

    def walk(o, path=()):
        if isinstance(o, dict):
            if isinstance(o.get("owns"), list):
                name = o.get("handle") or o.get("name") or (path[-1] if path else "?")
                pats.extend((str(p), str(name)) for p in o["owns"])
            for k, v in o.items():
                walk(v, path + (k,))
        elif isinstance(o, list):
            for v in o:
                walk(v, path)

    walk(reg)
    pats.sort(key=lambda x: -len(x[0]))

    def owner(f: str) -> str:
        for pat, name in pats:
            p = pat.rstrip("/")
            if f == p or f.startswith(p + "/") or fnmatch.fnmatch(f, pat):
                return name.lstrip("@")
        return "(none)"

    return owner


def spelling(line: str, old: str) -> str:
    """The whole identifier on ``line`` that holds ``old``, else ``old`` itself (it stands in prose)."""
    low = old.lower()
    hits = [m.group(0) for m in IDENT.finditer(line) if low.replace(" ", "") in m.group(0).lower().replace("_", "").replace("-", "")
            or low in m.group(0).lower()]
    hits = [h for h in hits if h.lower() != low]
    if not hits:
        return old
    # the shortest identifier that still holds the old name is the name itself, not the dotted path around it
    parts = []
    for h in hits:
        for piece in re.split(r"[./]", h):
            if low.replace(" ", "").replace("-", "").replace("_", "") in piece.lower().replace("_", "").replace("-", ""):
                parts.append(piece)
    return min(parts or hits, key=len)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/sweep")
    ap.add_argument("--term")
    ap.add_argument("--not-term", dest="not_term")
    ap.add_argument("--say")
    ap.add_argument("--not-say", dest="not_say")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = lint()
    owner = owners()
    keep = re.compile(a.term, re.I) if a.term else None
    drop = re.compile(a.not_term, re.I) if a.not_term else None
    say_keep = re.compile(a.say, re.I) if a.say else None
    say_drop = re.compile(a.not_say, re.I) if a.not_say else None
    cache: dict[str, list[str]] = {}
    rows = []
    for f, line, what in m.scan():
        mo = re.match(r"'(.+?)' is retired; say (.*) \(GLOSSARY\.md\)$", what, re.S)
        old, say = (mo.group(1), mo.group(2)) if mo else (what[:60], "?")
        old1 = " ".join(old.split())
        if keep and not keep.search(old1):
            continue
        if drop and drop.search(old1):
            continue
        say1 = " ".join(say.split())
        if say_keep and not say_keep.match(say1):
            continue
        if say_drop and say_drop.match(say1):
            continue
        if f not in cache:
            cache[f] = Path(f).read_text(encoding="utf-8", errors="replace").split("\n")
        text = cache[f][line - 1] if 0 < line <= len(cache[f]) else ""
        rows.append((f, line, old1, " ".join(say.split()), owner(f), spelling(text, old1)))
    with (out / "sites.tsv").open("w") as fh:
        fh.write("file\tline\told\tterm\towner\tspelling\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    names: dict[tuple[str, str], dict] = {}
    for f, line, old, say, own, sp in rows:
        d = names.setdefault((sp, say), {"sites": 0, "files": set(), "lean": 0, "own": collections.Counter(), "ex": []})
        d["sites"] += 1
        d["files"].add(f)
        d["lean"] += f.endswith(".lean")
        d["own"][own] += 1
        if len(d["ex"]) < 3:
            d["ex"].append(f"{f}:{line}")
    with (out / "names.tsv").open("w") as fh:
        fh.write("spelling\tterm\tsites\tfiles\tlean_sites\towners\texamples\tnew\tkind\tclass\tevidence\n")
        for (sp, say), d in sorted(names.items(), key=lambda kv: -kv[1]["sites"]):
            own = ", ".join(f"{o}:{n}" for o, n in d["own"].most_common())
            fh.write(f"{sp}\t{say}\t{d['sites']}\t{len(d['files'])}\t{d['lean']}\t{own}\t{' '.join(d['ex'])}\t\t\t\t\n")
    tab = collections.Counter((own, say[:40]) for _, _, _, say, own, _ in rows)
    with (out / "by_owner.tsv").open("w") as fh:
        fh.write("owner\tterm\tsites\n")
        for (own, say), n in sorted(tab.items(), key=lambda kv: (kv[0][0], -kv[1])):
            fh.write(f"{own}\t{say}\t{n}\n")
    print(f"sites {len(rows)}  files {len({r[0] for r in rows})}  distinct spellings {len(names)}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
