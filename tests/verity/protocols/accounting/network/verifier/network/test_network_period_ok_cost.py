"""The cost of the period check's ok rule (Lean's `okChecks`): the schedule's constants are hashed once a period,
and holding an execution phase's registration to them hashes nothing. Lean is read: the period check's module applies
`ScanOk.consts` once, outside the function it maps over the execution phases, and neither that function nor
`ScanOk.held` names a definition that hashes (`ScanOk.calls` and its guards count what `ScanOk.consts` hashes)."""

from __future__ import annotations

import re
from pathlib import Path

LEAN = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "accounting" / "network" / "verifier" / "lean" / "NetworkVerifier"
#: `/- … -/` and `-- …`, and string literals: what a definition's text says and doesn't run
SILENT = re.compile(r'/-.*?-/|--[^\n]*|"(?:\\.|[^"\\])*"', re.S)
#: where a top-level declaration starts
DECLARATION = re.compile(r"^(?:def|theorem|structure|abbrev|instance|namespace|end|#guard|@\[)", re.M)
#: a `#guard`, to the next top-level line: what the build evaluates and no check runs
GUARD = re.compile(r"^#guard.*?(?=^\S)", re.S | re.M)
#: an identifier with its qualifiers and field accesses, so a field (`c.root`) is not the definition (`root`)
IDENTIFIER = re.compile(r"[A-Za-z_][\w.']*")
#: a name of something that hashes, wherever it is defined
HASHES = re.compile(r"(?i)sha|hash|digest|fv3|hm96|registered|known")
#: `ScanOk.held` and the definitions of `ScanOk` it reaches: what compares a registration with the constants
COMPARES = {"held", "entryHeld", "rootsOf", "named", "counted", "okPort", "PORT", "START"}


def code(module: str, *defines: str) -> str:
    """The code that runs of the check's module `NetworkCheck.<module>`: comments and strings blanked, its guards cut.
    The module is named, not found by a definition's name, because two modules may each define one (`ScanOk.held`
    holds a registration to the constants; the period check's own `held` holds the class circuit). It opens the one
    namespace of its name, so `<module>.<name>` in another module is its definition `name`, and it defines each of
    `defines` once."""
    text = GUARD.sub("", SILENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), (LEAN / f"{module}.lean").read_text()))
    assert re.findall(r"^namespace (\S+)", text, re.M) == [f"NetworkVerifier.{module}"], module
    for name in defines:
        assert len(re.findall(rf"^def {re.escape(name)}\b", text, re.M)) == 1, (module, name)
    return text


def definition(text: str, name: str) -> str:
    """The definition `name`'s code in a module's: from its `def` to the next top-level declaration."""
    start = re.search(rf"^def {re.escape(name)}\b", text, re.M)
    assert start, name
    end = DECLARATION.search(text, start.end())
    return text[start.start():end.start() if end else len(text)]


def names(body: str) -> set[str]:
    return set(IDENTIFIER.findall(body))


def test_lean_applies_the_constants_once_outside_the_execution_phases():
    """The period check's module names `ScanOk.consts` once, in `okChecks`, as the argument of `okChecksAt`, and
    `okChecks` maps over nothing itself; `check` reads `okChecks` once. Lean evaluates an argument before the call, so
    the constants are computed once a `check`, whatever the registrations."""
    period = code("Period", "okChecks", "okChecksAt", "okCheck", "check")
    assert len(re.findall(r"\bScanOk\.consts\b", period)) == 1
    ok_checks = definition(period, "okChecks")
    assert re.search(r"=> okChecksAt \(ScanOk\.consts s\.window\) p\.first c\.windows\n", ok_checks), ok_checks
    assert not names(ok_checks) & {"fun", "for", "do"} and not re.search(r"\.map\b|\.mapM\b|<\$>", ok_checks)
    assert len(re.findall(r"\bokChecks\b", period)) == 2 and "(okChecks p c).toList" in definition(period, "check")
    assert len(re.findall(r"\bokChecksAt\b", period)) == 2 and len(re.findall(r"\bokCheck\b", period)) == 2


def test_lean_holds_an_execution_phase_by_comparison_alone():
    """What runs once an execution phase names nothing that hashes: `okChecksAt` and `okCheck` read the constants as a
    value and call `ScanOk.held`, never the period check's own `held`, and every definition of `ScanOk` that its
    `held` reaches (`COMPARES`) names, of `ScanOk`'s own, only another of them, and no definition that hashes (`salt`,
    `known`, `domain`, `root`, `consts`)."""
    period, ok = code("Period", "okChecksAt", "okCheck"), code("ScanOk", "held")
    own = set(re.findall(r"^def ([\w']+)", ok, re.M))
    assert {"salt", "known", "domain", "root", "consts", "members"} | COMPARES <= own
    reached, todo = set(), ["held"]
    while todo:
        name = todo.pop()
        if name in reached:
            continue
        reached.add(name)
        body = names(definition(ok, name))
        assert body & own <= COMPARES, (name, body & own)
        assert not [n for n in body if HASHES.search(n)], (name, body)
        todo += sorted(body & own)
    assert reached == COMPARES, reached ^ COMPARES
    for name in ("okChecksAt", "okCheck"):
        body = names(definition(period, name))
        assert {n for n in body if n.startswith("ScanOk.")} <= {"ScanOk.held", "ScanOk.Const"}, (name, body)
        assert "held" not in body, (name, body)
        assert not [n for n in body if HASHES.search(n)], (name, body)
    assert "okCheck" in names(definition(period, "okChecksAt")) and "ScanOk.held" in names(definition(period, "okCheck"))
