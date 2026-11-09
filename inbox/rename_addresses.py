#!/usr/bin/env python3
"""Rename SPOT[Flock]'s per-gate premise and the names after it, in the lines a branch adds over BASE.

    python3 rename_addresses.py BASE FILE...

Only lines that BASE's copy of the file doesn't hold are touched, so nothing on BASE moves. In Lean, code (outside
comments and strings) renames whole identifiers by CODE; in comments and docstrings, backtick spans rename by CODE and
the prose renames whole words by PROSE, with "a" before a renamed word made "an". A `.json` file's lines are prose with
backtick spans, and a dotted name in them renames by CODE only under `SpotFlock.`. A dotted name with a part in KEEP
(another area's namespace) is never renamed. Another pair of names costs a new CODE and PROSE.
"""
from __future__ import annotations

import re
import subprocess
import sys

CODE = {
    "Sites": "Addresses",
    "Site": "Address",
    "SiteHolds": "AddressHolds",
    "siteMap": "addressMap",
    "siteMap_spec": "addressMap_spec",
    "sitePlace": "addressPlace",
    "siteOp": "addressOp",
    "siteOp_seenAt": "addressOp_seenAt",
    "openAt_siteOp": "openAt_addressOp",
    "site": "addr",
}
PROSE = {"sites": "addresses", "Sites": "Addresses", "site": "address", "Site": "Address"}
KEEP = {"Core", "Verity", "HiddenExec", "Placed", "Integrate", "Layout", "FlockSoundness", "Properties",
        "CoreProtocol", "Discharge", "Definitions", "Proofs", "Soundness", "Registration"}

NAME = re.compile(r"(?<![\w'!?.])[A-Za-z_][\w'!?]*(?:\.[A-Za-z_][\w'!?]*)*")
WORD = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, PROSE), key=len, reverse=True)) + r")(?!\w)")
TICKS = re.compile(r"(`+)(.+?)\1")
ARTICLE = re.compile(r"(?<!\w)([Aa]) (?=`*(?:[Aa]ddress|addr)\b)")


def code(text: str, json: bool = False) -> str:
    def one(m: re.Match) -> str:
        parts = m.group(0).split(".")
        if set(parts) & KEEP or (json and len(parts) > 1 and parts[0] != "SpotFlock"):
            return m.group(0)
        return ".".join(CODE.get(p, p) for p in parts)
    return NAME.sub(one, text)


def prose(text: str, json: bool = False) -> str:
    out, at = [], 0
    for m in TICKS.finditer(text):
        out += [words(text[at:m.start()], json), m.group(1) + code(m.group(2), json) + m.group(1)]
        at = m.end()
    out.append(words(text[at:], json))
    return ARTICLE.sub(lambda m: m.group(1) + "n ", "".join(out))


def words(text: str, json: bool) -> str:
    """PROSE over ``text``, a dotted name in it renamed as code (under `SpotFlock.` alone in JSON)."""
    out, at = [], 0
    for m in re.finditer(r"[A-Za-z_][\w']*(?:\.[A-Za-z_][\w']*)+", text):
        out += [WORD.sub(lambda w: PROSE[w.group(1)], text[at:m.start()]), code(m.group(0), json)]
        at = m.end()
    out.append(WORD.sub(lambda w: PROSE[w.group(1)], text[at:]))
    return "".join(out)


def lean_classes(src: str) -> list[str]:
    """Each character's class: "c" code, "s" string, "p" comment or docstring."""
    cls, i, depth, n = [""] * len(src), 0, 0, len(src)
    while i < n:
        if depth:
            if src.startswith("/-", i):
                depth, step = depth + 1, 2
            elif src.startswith("-/", i):
                depth, step = depth - 1, 2
            else:
                step = 1
            cls[i:i + step] = ["p"] * step
            i += step
        elif src.startswith("/-", i):
            depth = 1
            cls[i:i + 2] = ["p", "p"]
            i += 2
        elif src.startswith("--", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            cls[i:j] = ["p"] * (j - i)
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            cls[i:j + 1] = ["s"] * (j + 1 - i)
            i = j + 1
        else:
            cls[i] = "c"
            i += 1
    return cls


def rename(path: str, base: str) -> int:
    src = open(path, encoding="utf-8").read()
    old = subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, text=True)
    kept = set(old.stdout.split("\n")) if old.returncode == 0 else set()
    is_json = path.endswith(".json")
    cls = None if is_json else lean_classes(src)
    out, at, changed = [], 0, 0
    for line in src.split("\n"):
        end = at + len(line)
        new = line
        if line not in kept:
            if is_json:
                new = prose(line, json=True)
            else:
                runs, k = [], at
                while k < end:
                    c, j = cls[k], k
                    while j < end and cls[j] == c:
                        j += 1
                    seg = src[k:j]
                    runs.append(code(seg) if c == "c" else prose(seg) if c == "p" else seg)
                    k = j
                new = "".join(runs)
        changed += new != line
        out.append(new)
        at = end + 1
    if changed:
        open(path, "w", encoding="utf-8").write("\n".join(out))
    return changed


if __name__ == "__main__":
    base, *paths = sys.argv[1:]
    for p in paths:
        print(f"{p}: {rename(p, base)} line(s)")
