"""Every repository path a tracked shell script, or a shell line a Python script builds, puts on `PYTHONPATH` exists.

A dead entry fails the suite read guard: `importlib.metadata` opens a missing entry as a zip, and the guard counts that open
as a read outside the suite's inputs, so a pod script that keeps a path the tree no longer has breaks every suite it runs.
An entry is checked when it is `$REPO`, `$SRC`, `$PWD`, `$T`, `$ROOT` or `$TREE` (braced or not) followed by a path with no
further `$`, `{` or `}` (a `${PYTHONPATH:-...}` default is not read); it exists when it resolves from the root. One that climbs with `..` resolves from the script's directory or one of its
ancestors instead, since that script sets its variable to its own package directory.
"""
from __future__ import annotations

import posixpath
import re

from test_repository import ROOT, _tracked

_ASSIGN = re.compile(r"PYTHONPATH=([^\s;&|]+)")
_ENTRY = re.compile(r"^\$(?:\{(?:REPO|SRC|PWD|T|ROOT|TREE)\}|(?:REPO|SRC|PWD|T|ROOT|TREE))/([^${}]+)$")


def _resolves(script: str, rel: str) -> bool:
    parts = script.split("/")[:-1]
    bases = ["/".join(parts[:i]) for i in range(len(parts), -1, -1)] if rel.startswith("..") else [""]
    for base in bases:
        path = posixpath.normpath(posixpath.join(base, rel.rstrip("/")))
        if not path.startswith("..") and (ROOT / path).exists():
            return True
    return False


def test_pythonpath_entries_exist() -> None:
    dead = []
    for script in _tracked():
        if not script.endswith((".sh", ".py")):
            continue
        for value in _ASSIGN.findall((ROOT / script).read_text(errors="replace")):
            for entry in value.strip("\"'").split(":"):
                m = _ENTRY.match(entry)
                if m and not _resolves(script, m.group(1)):
                    dead.append(f"{script}: {entry}")
    assert not dead, "PYTHONPATH entries that name no path in the tree; drop them:\n" + "\n".join(dead)
