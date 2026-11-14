"""The cost of the period check's ok rule (Lean's `okChecks`, the mirror's `scan_ok` calls): the schedule's constants
are hashed once a period, the same number of SHA-512 calls whatever the period's length, and holding an execution
phase's registration to them hashes nothing.

The mirror is counted as it runs: every SHA-512 call, through the commitments' registry and through hashlib, the two
names the verifier's Python hashes by, over periods of one, two and three execution phases under schedules of one T,
the one accepted and the longer ones refused for being longer and for nothing else. Lean is read: the period check's module applies `ScanOk.consts` once, outside the function it maps over the
execution phases, and neither that function nor `ScanOk.held` names a definition that hashes (`ScanOk.calls` and its
guards count what `ScanOk.consts` hashes)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import period_case as PC
from verity.commitments import scheme
from verity.protocols.accounting.certifier.verifier import scan_ok

LEAN = Path(__file__).resolve().parents[2] / "lean" / "NetworkCheck"
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
    assert re.findall(r"^namespace (\S+)", text, re.M) == [f"NetworkCheck.{module}"], module
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


def periods(monkeypatch) -> dict[int, dict]:
    """A rows-form period of each length, one to three execution phases, each under a schedule of `PC.EQUAL_T` ticks:
    the run's first execution phase alone, accepted, and it with the next one or two, each registration what an accepted
    one's is and the period refused for holding more than the first (`PC.CARRIED`)."""
    real = PC._small_schedule
    monkeypatch.setattr(PC, "_small_schedule", lambda r, small, capped=False, T=None: real(r, small, capped, PC.EQUAL_T))
    monkeypatch.setattr(PC, "CARRIES", ("more",))
    out: dict[int, dict] = {1: PC.build("rows-ok-equal", 0)}
    for seed in range(64):
        case = PC.build("rows-start-carried", seed)
        out.setdefault(case["params"]["windows"], case)
        if len(out) == 3:
            break
    assert sorted(out) == [1, 2, 3], sorted(out)
    assert {c["params"]["schedule"]["window"] for c in out.values()} == {PC.EQUAL_T}
    return out


@pytest.fixture
def spans(monkeypatch):
    """`scan_ok.constants` and `scan_ok.held` as the mirror calls them, each call's SHA-512 calls recorded."""
    calls, real, out = [], hashlib.sha512, {"constants": [], "held": []}

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    def spanned(name, f):
        def g(*args, **kwargs):
            before = len(calls)
            try:
                return f(*args, **kwargs)
            finally:
                out[name].append(len(calls) - before)
        return g

    monkeypatch.setitem(scheme._SHA, "sha512", counting)
    monkeypatch.setattr(hashlib, "sha512", counting)
    monkeypatch.setattr(scan_ok, "constants", spanned("constants", scan_ok.constants))
    monkeypatch.setattr(scan_ok, "held", spanned("held", scan_ok.held))
    return out


def test_the_mirror_hashes_the_constants_once_a_period_and_nothing_an_execution_phase(spans, monkeypatch):
    """One call of the constants a period, of the same SHA-512 calls at one, two and three execution phases, and one
    comparison an execution phase that hashes nothing: so a period of m execution phases costs the ok rule what a
    period of one does."""
    cases = periods(monkeypatch)
    once = set()
    for m, case in sorted(cases.items()):
        for span in spans.values():
            span.clear()
        got = PC.reference(case)
        assert got["verdict"] == PC.ACCEPT if m == 1 else PC.CARRIED in got["reason"], (m, got)
        assert len(spans["constants"]) == 1 and spans["held"] == [0] * m, (m, spans)
        once.add(spans["constants"][0])
    assert len(once) == 1 and min(once) > 0, once


def test_a_refused_period_costs_the_same(spans):
    """A period refused at its ok bits hashed the constants once and compared every execution phase's registration,
    hashing nothing: the refusal is no cheaper or dearer than the acceptance."""
    for seed in range(12):
        case = PC.build("rows-ok-unheld", seed)
        for span in spans.values():
            span.clear()
        out = PC.reference(case)
        assert out["verdict"] == PC.REJECT and PC.UNOK in out["reason"], out
        assert len(spans["constants"]) == 1 and spans["held"] == [0] * case["params"]["windows"], spans


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
