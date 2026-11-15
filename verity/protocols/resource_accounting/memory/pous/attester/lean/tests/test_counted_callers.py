"""No executable of this package serves the wipe's counted check but the difftest's own (`wipe-count-test`).

`MemoryChallenger.WipeProtocol.checkCounted` certifies at the count the evidence states, and nothing holds that count to
the network certifier's count over the span until a composition ties `named` to it. So an executable that reaches the
check (`checkCounted`, or a definition of `MemoryChallenger.WipeCount` built on it: `mkCounted`, `countedTest`) other than
the ones `ALLOWED` names fails here: each `lean_exe` of `lakefile.toml`, its root module and every module of this package
its `import` lines reach, read with comments left out. `named` alone, which `memory-challenger-difftest` reads, is the
tie's half and serves nothing. The Python mirror's callers are held in the repository's suite
(`tests/test_wipe_counted_callers.py`). Reads files only: no build, no clock."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
WIPE_COUNT = "MemoryChallenger.WipeCount"
#: The definition every counted one is built on.
SEED = "checkCounted"
#: The executables that may reach the counted check, each with why.
ALLOWED: dict[str, str] = {
    "wipe-count-test": "the difftest's port of the counted check, under the test signers' schemes alone",
}
REMEDY = ("the counted check certifies on a count nobody holds until a composition ties `named` to the network "
          "certifier's count: only that composition may serve it, and this test's `ALLOWED` names its executable then")

COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)
IMPORT = re.compile(r"^(?:(?:public|private|meta)\s+)*import\s+(?:all\s+)?([\w.]+)", re.M)
DECL = re.compile(r"(?:(?:public|private|protected|noncomputable|partial|@\[[^\]]*\])\s+)*(?:def|abbrev|instance|theorem)\s+"
                  r"([\w.]+)")
#: A top-level command starts at a line's first column; what continues it is indented.
TOP = re.compile(r"^(?=\S)", re.M)


def _code(module: str) -> str | None:
    path = PACKAGE / (module.replace(".", "/") + ".lean")
    return COMMENT.sub("", path.read_text()) if path.is_file() else None


def _names(text: str, names: set[str]) -> set[str]:
    return {n for n in names if re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", text)}


def counted() -> set[str]:
    """`SEED` and every definition of `WipeCount` whose body names one already in the set."""
    code = _code(WIPE_COUNT)
    assert code is not None, f"{WIPE_COUNT} is gone: point this test at the module that defines `{SEED}`"
    decls = {m.group(1).split(".")[-1]: block[m.end():]
             for block in TOP.split(code) if (m := DECL.match(block))}
    assert SEED in decls, f"{WIPE_COUNT} defines no `{SEED}`: point this test at the counted check's new name"
    found = {SEED}
    while grown := {d for d, body in decls.items() if d not in found and _names(body, found)}:
        found |= grown
    return found


def reached(root: str) -> list[str]:
    """`root` and every module of this package its imports reach, `root` first."""
    seen, todo = [], [root]
    while todo:
        module = todo.pop()
        if module in seen or (code := _code(module)) is None:
            continue
        seen.append(module)
        todo.extend(IMPORT.findall(code))
    return seen


def serves(root: str, names: set[str]) -> dict[str, set[str]]:
    """Each module `root` reaches, `WipeCount` aside, with the counted definitions its code names."""
    return {m: hit for m in reached(root) if m != WIPE_COUNT and (hit := _names(_code(m) or "", names))}


def executables() -> dict[str, str]:
    lake = tomllib.loads((PACKAGE / "lakefile.toml").read_text())
    return {e["name"]: e["root"] for e in lake.get("lean_exe", [])}


def test_the_counted_definitions_are_found() -> None:
    assert {"checkCounted", "mkCounted", "countedTest"} <= counted()


def test_no_executable_but_the_allowed_ones_serves_the_counted_check() -> None:
    names = counted()
    offenders = {exe: hits for exe, root in executables().items() if exe not in ALLOWED and (hits := serves(root, names))}
    assert not offenders, "\n".join(
        [f"`{exe}` reaches the counted check: " + "; ".join(f"{m} names {', '.join(sorted(h))}" for m, h in hits.items())
         for exe, hits in offenders.items()] + [f"Remove it: {REMEDY}."])


def test_every_allowed_executable_still_serves_it() -> None:
    names, exes = counted(), executables()
    stale = [exe for exe in ALLOWED if exe not in exes or not serves(exes[exe], names)]
    assert not stale, f"ALLOWED names {stale}, which no longer serve the counted check: delete the entries"


def test_a_use_is_caught_and_a_comment_is_not() -> None:
    code = "/-- `checkCounted` -/\n-- countedTest\ndef main := WipeProtocol.mkCounted \"id\" SCHEMES\ndef mkCounted' := 0"
    assert _names(COMMENT.sub("", code), counted()) == {"mkCounted"}
