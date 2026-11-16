"""What the first pass added to each old word: the Glossary row that names it, the part of it a row retires, the other live
sense, and where an old spelling is held by a stored record."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

older: list[str] = []
_on = False
for _line in Path("GLOSSARY.md").read_text(encoding="utf-8").split("\n"):
    if _line.startswith("## "):
        _on = _line.strip() == "## Older names"
    elif _on and _line.startswith("|") and not _line.startswith("| ---") and not _line.startswith("| Old"):
        older.append(_line.strip("|").split(" | ")[0].lower())

TOKENS: dict[str, str] = {}
for _cell in older:
    for _a, _b in re.findall(r'"([^"]+)"|`([^`]+)`', _cell):
        TOKENS.setdefault((_a or _b).strip(), _cell)
SINGLE = {k.lower() for k in TOKENS if re.fullmatch(r"[A-Za-z]+", k)}


def names_it(old: str) -> str:
    o = old.lower()
    for tok in TOKENS:
        if tok == o or tok.rstrip("*") == o.rstrip("*") or (tok.endswith("*") and o.startswith(tok.rstrip("*"))):
            return tok
    return ""


def part_named(old: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_ -]+", old):
        return ""
    ps = re.findall(r"[a-z0-9]+", HUMP.sub("_", old).lower())
    return ",".join(p for p in ps if p in SINGLE)


JUDGED = {
    "registration": "the registration log, the `registration` protocol and the registered values keep the word (AGENTS.md); the map renamed it only where it meant the docket",
    "registrations": "same as registration (plural)",
    "registration record": "same as registration; the registration log is the live phrase",
    "drew": "ordinary English past tense, and a person's name in a few files; the map kept those",
    "undrawn": "ordinary English; the map kept a [keep] pattern for it",
    "redrawn": "ordinary English (figures, plots) beside the sampling sense",
    "redraws": "ordinary English (figures, plots) beside the sampling sense",
    "novel": "ordinary English adjective (a novel result)",
    "epoch": "a training epoch, and the time base of a timestamp, beside the execution phase sense",
    "auditor host": "the auditor's host is a kind of location in verity/device.py (KINDS)",
    "sample": "the Glossary's own word (a sample of proof blocks); the map renamed only the budget's plan point",
}

_keeps: dict = {}


def _git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, errors="replace").stdout


def _map_data(mapname: str, commit: str) -> dict:
    txt = _git("show", f"{commit}:tools/move/{mapname}") if commit and commit != "worktree" else ""
    if not txt:
        p = Path("tools/move") / mapname
        txt = p.read_text() if p.exists() else ""
    import maps_words
    try:
        return maps_words.parse(txt) or {}
    except Exception:
        return {}


def keeps_of(mapname: str, commit: str):
    """The regexes a map's [keep] holds (its `patterns`, and its `strings` escaped), as the move tool reads them."""
    key = (mapname, commit)
    if key not in _keeps:
        keep = _map_data(mapname, commit).get("keep", {})
        _keeps[key] = list(keep.get("patterns", [])) + [re.escape(s) for s in keep.get("strings", [])]
    return _keeps[key]


def keeps_in_of(mapname: str, commit: str):
    """(glob, word) for each word [keep_in] or [keep_paths] lists for the files a glob matches."""
    d = _map_data(mapname, commit)
    return [(g, w) for sec in ("keep_in", "keep_paths") for g, ws in d.get(sec, {}).items() for w in ws]


def live_sense(old: str, pairs: list[dict], scoped: bool) -> str:
    why = []
    if old.lower() in JUDGED:
        why.append(JUDGED[old.lower()])
    if scoped:
        why.append("the map renames it only in listed files; elsewhere it is another word")
    hits = []
    for p in pairs:
        for pat in keeps_of(p["map"], p["commit"]):
            if old.lower() in re.sub(r"[\\\[\]()?*+^$]|\(\?[^)]*\)", "", pat).lower():
                hits.append(f"{p['map']}: {pat[:60]}")
                break
        else:
            for g, w in keeps_in_of(p["map"], p["commit"]):
                if old.lower() in w.lower():
                    hits.append(f"{p['map']}: [keep_in] {g}: {w}")
                    break
    if hits:
        why.append("the map's [keep] holds it: " + "; ".join(hits[:2]))
    return " | ".join(why) if why else "no sign (not scoped, no [keep] pattern holds it)"


def reading(path: str, text: str) -> str:
    """A first guess at what holds an old spelling at a site; the sampled sites are read by hand and overrule it."""
    if path.endswith("moves.json"):
        return "held: the store's moves.json records it"
    if path.endswith("lean-audit.json"):
        return "held: a lock record"
    if "/fixtures/" in path or path.endswith((".json", ".jsonl", ".bin", ".lock")):
        return "held?: a data file or fixture a reader holds (read to confirm)"
    return ""
