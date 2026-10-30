"""The failures `main` already has, each with its cause: `circuit-check --all` and the suite pass on them and fail on anything
else, and on an entry that no longer fails (a fixed one is dropped here).

Each is a file of its own (`entries`) under ``known/``, with its ``key`` and ``reason``: ``known/failures/`` holds a failure by
``"<check>/<code> <Definition family>"`` (`key`), ``known/pieces/`` a piece of the Flock lowering that differs from its
primitive, by its id (a Definition lowered through one inherits it).  An entry's file is ``entries.name(key) + ".toml"``.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Iterable

from . import entries as E

DIR = Path(__file__).with_name("known")


def load(directory: Path) -> dict[str, str]:
    """{key: reason} of every entry file in ``directory``; one with other fields, or not named by its key, is an error."""
    out: dict[str, str] = {}
    for p in E.files(directory, ".toml"):
        d = tomllib.loads(p.read_text())
        if set(d) != {"key", "reason"} or not all(isinstance(v, str) for v in d.values()):
            raise ValueError(f"{p}: a known entry is a key and a reason, both strings")
        E.check(p, d["key"])
        out[d["key"]] = d["reason"]
    return dict(sorted(out.items()))


#: pieces of the Flock lowering that differ from their primitive, and why; a Definition lowered through one inherits it
KNOWN_PIECES: dict[str, str] = load(DIR / "pieces")
#: other failures main has, by "<check>/<code> <Definition family>", and why
KNOWN: dict[str, str] = load(DIR / "failures")


def key(f: dict) -> str:
    return f"{f['check']}/{f['code']} {f['definition'].split('{')[0]}"


def known(f: dict) -> str | None:
    """Why the failure ``f`` is known, or None when it is new."""
    if key(f) in KNOWN:
        return KNOWN[key(f)]
    if f["code"] == "lowering-mismatch":
        causes = [f["definition"]] if f["definition"] in KNOWN_PIECES else f.get("causes") or []
        if causes and all(c in KNOWN_PIECES for c in causes):
            return "; ".join(f"{c}: {KNOWN_PIECES[c]}" for c in causes)
    return None


def stale(failures: Iterable[dict]) -> list[str]:
    """The entries no failure among ``failures`` (every registered Definition's) matches any more."""
    fs = list(failures)
    keys = {key(f) for f in fs}
    pieces = {f["definition"] for f in fs if f["code"] == "lowering-mismatch"}
    return sorted(set(KNOWN) - keys) + sorted(set(KNOWN_PIECES) - pieces)


__all__ = ["DIR", "KNOWN", "KNOWN_PIECES", "key", "known", "load", "stale"]
