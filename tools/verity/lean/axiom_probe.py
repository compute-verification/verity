"""The axiom probe `verity lean try` runs: lean's exit 0 says a file elaborates, not that its theorems are proved, since one
resting on `sorry` (`sorryAx`) or an `axiom` elaborates too.  So `try` elaborates a copy of FILE with `#print axioms N`
appended for each of its top-level theorems (`theorems`: a scan of its code, comments and strings aside, that tracks
`namespace`, `section`, `mutual` and `end`), in one elaboration, and its last line (`summary`) names each theorem whose
axioms go beyond `propext`, `Classical.choice` and `Quot.sound`, or says none do.  An error on an appended line is the
probe's own (`own_rc`): the file's status is lean's on its own lines.  Stdlib only: `lean_changed.py --try` and the CLI's
local `verity lean try` both load it."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

#: the axioms a proved theorem may use, as the Lean check allows them
STANDARD = ("propext", "Classical.choice", "Quot.sound")
#: where in its package a probed copy goes: inside `.lake`, which the build sandbox leaves writable and `sync` leaves alone
COPIES = Path(".lake") / "axiom-probe"
MODIFIERS = r"(?:(?:public|private|protected|noncomputable|nonrec|unsafe|partial)\s+)*"
NAMESPACE = re.compile(rf"^\s*{MODIFIERS}namespace\s+(\S+)")
SECTION = re.compile(rf"^\s*(?:@\[[^\]]*\]\s*)?{MODIFIERS}(?:section|mutual)\b")
END = re.compile(r"^\s*end\b")
THEOREM = re.compile(rf"^\s*(?:@\[[^\]]*\]\s*)*{MODIFIERS}(?:theorem|lemma)\s+([^\s:(){{}}\[\]⦃]+)")
#: `#print axioms`' answer; a private declaration's name comes as `_private.MODULE.0.NAME` (in a module, every one not public)
ANSWER = re.compile(r"^'(.+?)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", re.M | re.S)
PRIVATE = re.compile(r"^_private\..*?\.0\.")
#: a Lean error's position, as lean prints it (`F:L:C: error`) and as Lake does (`error: F:L:C:`)
ERROR_AT = re.compile(r"^(?:error: )?(?:\./)*(\S+?\.lean):(\d+):\d+: error", re.M)


def code(text: str) -> str:
    """``text`` with its comments (`--`, nested `/- -/`, docstrings included) and string literals blanked, lines kept."""
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        c, two = text[i], text[i:i + 2]
        if depth:
            depth += (two == "/-") - (two == "-/")
            out.append(c if c == "\n" else " ")
            if two in ("/-", "-/"):
                out.append(" ")
                i += 1
        elif two == "/-":
            depth, i = 1, i + 1
            out.append("  ")
        elif two == "--":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        elif text[i:i + 3] == "'\"'":
            out.append("   ")
            i += 2
        elif c == '"':
            out.append(" ")
            i += 1
            while i < n and text[i] != '"':
                step = 2 if text[i] == "\\" else 1
                out += [s if s == "\n" else " " for s in text[i:i + step]]
                i += step
            out.append(" ")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def theorems(text: str) -> list[str]:
    """The full names of ``text``'s top-level theorems (and lemmas), in order: each under the namespaces open where it is
    declared, a `_root_.` name as it is."""
    scopes: list[list[str]] = []
    names = []
    for line in code(text).splitlines():
        if m := NAMESPACE.match(line):
            scopes.append(m.group(1).split("."))
        elif SECTION.match(line):
            scopes.append([])
        elif END.match(line):
            if scopes:
                scopes.pop()
        elif m := THEOREM.match(line):
            name = m.group(1)
            names.append(name.removeprefix("_root_.") if name.startswith("_root_.") else
                         ".".join([*(p for s in scopes for p in s), name]))
    return list(dict.fromkeys(names))


def with_probe(text: str, names: list[str]) -> str:
    """``text`` with a `#print axioms` line for each of ``names`` after its last line."""
    return text + ("" if text.endswith("\n") else "\n") + "\n" + "".join(f"#print axioms {n}\n" for n in names)


def own_rc(rc: int, output: str, lines: int, copy: str) -> int:
    """The file's own status from lean's ``rc`` on its probed ``copy``: 0 when every error lean reports is on a line of
    ``copy`` past the file's ``lines``, one the probe appended."""
    at = [(m.group(1), int(m.group(2))) for m in ERROR_AT.finditer(output)]
    return 0 if rc and at and all(f == copy and n > lines for f, n in at) else rc


def axioms(output: str) -> dict[str, list[str]]:
    """{theorem: its axioms} from `#print axioms`' answers in ``output``."""
    return {PRIVATE.sub("", m.group(1)): [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
            for m in ANSWER.finditer(output)}


def summary(names: list[str], output: str) -> str:
    """The probe's line: each of ``names`` whose axioms go beyond `STANDARD`, with them, and those it got no answer for."""
    if not names:
        return "axiom probe: no theorems in it"
    got = axioms(output)
    beyond = {n: [a for a in got[n] if a not in STANDARD] for n in names if n in got}
    bad = [f"{n} ({', '.join(a)})" for n, a in beyond.items() if a]
    unread = [n for n in names if n not in got]
    std = ", ".join(STANDARD[:-1]) + f" and {STANDARD[-1]}"
    of = f"the {len(beyond)} it read" if unread else f"its {len(names)} theorems"
    said = (f"axiom probe: {len(bad)} of {of} use axioms beyond {std}: {'; '.join(bad)}" if bad else
            f"axiom probe: none of {of} uses an axiom beyond {std}")
    return said + (f"; no answer for {len(unread)} (an error on its `#print axioms` line above): {', '.join(unread)}"
                   if unread else "")


def probe(pkg: Path, rel: str, text: str, lean: Callable[[str], tuple[int, str]]) -> tuple[int, str, str]:
    """(the file's status, the probe's line, lean's output) for ``text``, the file at ``rel`` in package ``pkg``: its copy
    with the probe appended, written at `COPIES`/``rel`` under ``pkg`` and elaborated by ``lean`` (given that path from
    ``pkg``; lean's (status, output)), then removed.  No line when the file's own lines fail: their errors are the report."""
    names = theorems(text)
    copy = COPIES / rel
    (pkg / copy).parent.mkdir(parents=True, exist_ok=True)
    (pkg / copy).write_text(with_probe(text, names))
    try:
        rc, output = lean(str(copy))
    finally:
        (pkg / copy).unlink(missing_ok=True)
    rc = own_rc(rc, output, len(text.splitlines()), str(copy))
    return rc, "" if rc else summary(names, output), output
