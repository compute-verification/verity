"""The axiom probe `verity lean try` runs: lean's exit 0 says a file elaborates, not that its theorems are proved, since one
resting on `sorry` (`sorryAx`) or an `axiom` elaborates too.  So `try` elaborates a copy of FILE with `#print axioms N`
appended for each of its top-level theorems (`theorems`: a scan of its code, comments and strings aside, that tracks
`namespace`, `section`, `mutual` and `end`), in one elaboration, and its last line (`summary`) names each theorem whose
axioms go beyond `propext`, `Classical.choice` and `Quot.sound`, or says none do.  An error on an appended line is the
probe's own (`own_rc`): the file's status is lean's on its own lines.  Where those fail, none of `#print axioms`' answers
is shown (`probe`), and the line names each theorem with an error of Lean's on its own lines (`spans`, `failed`).  An
answer is no report on a theorem of a file lean refuses: `#print axioms` takes a name the elaborator knows and reads its
axioms from what the kernel holds, so for a theorem the kernel never got (a `rfl` that ran out of recursion depth, say) it
says "does not depend on any axioms".
Stdlib only: `lean_changed.py --try` and the CLI's local `verity lean try` both load it."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

#: the axioms a proved theorem may use, as the Lean check allows them
STANDARD = ("propext", "Classical.choice", "Quot.sound")
#: where in its package a probed copy goes: inside `.lake`, which the build sandbox leaves writable and `sync` leaves alone
COPIES = Path(".lake") / "axiom-probe"
MODIFIER = r"(?:public|private|protected|noncomputable|nonrec|unsafe|partial)"
MODIFIERS = rf"(?:{MODIFIER}\s+)*"
NAMESPACE = re.compile(rf"^\s*{MODIFIERS}namespace\s+(\S+)")
SECTION = re.compile(rf"^\s*(?:@\[[^\]]*\]\s*)?{MODIFIERS}(?:section|mutual)\b")
END = re.compile(r"^\s*end\b")
THEOREM = re.compile(rf"^\s*(?:@\[[^\]]*\]\s*)*{MODIFIERS}(?:theorem|lemma)\s+([^\s:(){{}}\[\]⦃]+)")
#: a line of code above a theorem that is part of it: attributes, a `… in` or modifiers, alone on their line
OPENS = re.compile(rf"^\s*(?:(?:@\[[^\]]*\]\s*)+|.*\bin|(?:{MODIFIER}\s*)+)\s*$")
#: a line of code at a theorem's indent, or left of it, that is still part of it
CONTINUES = re.compile(r"^\s*(?:[|)\]}⟩]|:=|(?:by|where|termination_by|decreasing_by)\b)")
#: `#print axioms`' answer; a private declaration's name comes as `_private.MODULE.0.NAME` (in a module, every one not public)
ANSWER = re.compile(r"^'(.+?)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)\n?", re.M | re.S)
#: the line an answer starts on (its list of axioms may run over the next ones)
ANSWERS = re.compile(r"'.+?' (?:depends on axioms: \[|does not depend on any axioms)")
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


def spans(text: str) -> list[tuple[str, int, int]]:
    """``text``'s top-level theorems (and lemmas), in order, each as (its full name, the first of its own lines, the last),
    lines counted from 1.  The name is under the namespaces open where it is declared, a `_root_.` name as it is.  Its own
    lines start under the last line of code above it that isn't an attribute, a `… in` or a modifier alone on its line
    (`OPENS`), so its docstring is among them: Lean puts an error on the whole declaration at its first line.  They end
    at its last line of code before the next theorem, `namespace`, `section`, `mutual` or `end`, or before a line at its
    indent or left of it that doesn't continue it (`CONTINUES`).  This scans the text and is not Lean's parse: an error Lean
    puts on a line the scan gives no theorem (the first line of an attribute list that runs over several lines, say) names
    none.  That costs the theorem's name in the probe's line and nothing else: no answer is shown for a file that fails
    (`probe`)."""
    scopes: list[list[str]] = []
    found = []
    raw, lines = text.splitlines(), code(text).splitlines()
    for i, line in enumerate(lines):
        if m := NAMESPACE.match(line):
            scopes.append(m.group(1).split("."))
        elif SECTION.match(line):
            scopes.append([])
        elif END.match(line):
            if scopes:
                scopes.pop()
        elif m := THEOREM.match(line):
            name = m.group(1)
            first, last, indent = i, i, len(raw[i]) - len(raw[i].lstrip())
            while first and (not lines[first - 1].strip() or OPENS.match(lines[first - 1])):
                first -= 1
            for j in range(i + 1, len(lines)):
                under = lines[j]
                if not under.strip():
                    continue
                if any(r.match(under) for r in (NAMESPACE, SECTION, END, THEOREM)) or (
                        len(under) - len(under.lstrip()) <= indent and not CONTINUES.match(under)):
                    break
                last = j
            found.append((name.removeprefix("_root_.") if name.startswith("_root_.") else
                          ".".join([*(p for s in scopes for p in s), name]), first + 1, last + 1))
    return found


def theorems(text: str) -> list[str]:
    """The full names of ``text``'s top-level theorems (and lemmas), in order (`spans`)."""
    return list(dict.fromkeys(name for name, _, _ in spans(text)))


def with_probe(text: str, names: list[str]) -> str:
    """``text`` with a `#print axioms` line for each of ``names`` after its last line."""
    return text + ("" if text.endswith("\n") else "\n") + "\n" + "".join(f"#print axioms {n}\n" for n in names)


def errors(output: str) -> list[tuple[str, int]]:
    """(file, line) of each error lean reports in ``output``."""
    return [(m.group(1), int(m.group(2))) for m in ERROR_AT.finditer(output)]


def own_rc(rc: int, output: str, lines: int, copy: str) -> int:
    """The file's own status from lean's ``rc`` on its probed ``copy``: 0 when every error lean reports is on a line of
    ``copy`` past the file's ``lines``, one the probe appended."""
    at = errors(output)
    return 0 if rc and at and all(f == copy and n > lines for f, n in at) else rc


def failed(found: list[tuple[str, int, int]], output: str, copy: str) -> list[str]:
    """The theorems of ``found`` (`spans`) that lean's ``output`` has an error for on their own lines of ``copy``."""
    at = {n for f, n in errors(output) if f == copy}
    return list(dict.fromkeys(name for name, first, last in found if any(first <= n <= last for n in at)))


def unanswered(output: str) -> str:
    """``output`` less every answer of `#print axioms`."""
    return ANSWER.sub("", output)


def axioms(output: str) -> dict[str, list[str]]:
    """{theorem: its axioms} from `#print axioms`' answers in ``output``."""
    return {PRIVATE.sub("", m.group(1)): [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
            for m in ANSWER.finditer(output)}


def summary(names: list[str], output: str, failures: list[str] | None = None) -> str:
    """The probe's line: each of ``names`` whose axioms go beyond `STANDARD`, with them, and those it got no answer for;
    or, when some failed (``failures``), those alone: the axioms of a file lean refuses are not reported."""
    if not names:
        return "axiom probe: no theorems in it"
    if failures:
        one = len(failures) == 1
        return (f"axiom probe: {len(failures)} of its {len(names)} theorems failed, with Lean's error{'' if one else 's'} "
                f"above on {'its' if one else 'their'} own lines: {', '.join(failures)}; no axioms are shown for a file "
                f"lean refuses")
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


def probe(pkg: Path, rel: str, text: str,
          lean: Callable[[str, Callable[[str], None]], tuple[int, str]]) -> tuple[int, str, str]:
    """(the file's status, the probe's line, lean's output) for ``text``, the file at ``rel`` in package ``pkg``: its copy
    with the probe appended, written at `COPIES`/``rel`` under ``pkg`` and elaborated by ``lean`` (given that path from
    ``pkg`` and where to hand each line of lean's output as it comes; lean's (status, output)), then removed.  It prints
    lean's lines as they come up to `#print axioms`' first answer, and the rest once lean is done: as it is when the file's
    own status is 0, and less every answer when its own lines fail, since an answer for a theorem of a file lean refuses
    can read as clean and mean nothing.  The line then names the theorems that failed (`failed`), and there is none when
    the scan finds no such theorem: the errors are the report."""
    found = spans(text)
    names = list(dict.fromkeys(name for name, _, _ in found))
    copy = COPIES / rel
    (pkg / copy).parent.mkdir(parents=True, exist_ok=True)
    (pkg / copy).write_text(with_probe(text, names))
    answers: list[str] = []

    def show(line: str) -> None:
        if answers or ANSWERS.match(line):
            answers.append(line)
        else:
            print(line, end="", flush=True)

    try:
        rc, output = lean(str(copy), show)
    finally:
        (pkg / copy).unlink(missing_ok=True)
    rc = own_rc(rc, output, len(text.splitlines()), str(copy))
    failures = failed(found, output, str(copy))
    rest = "".join(answers)
    print(unanswered(rest) if rc else rest, end="", flush=True)
    return rc, summary(names, output, failures) if failures or not rc else "", output
