"""No Python calls the wipe's counted check (`wipe_period.outcome_counted`, `counted_profile`) but its own module and the
memory challenger's Lean package's tests, which hold it to `MemoryChallenger.WipeCount`.

`outcome_counted` certifies at the count the evidence states, and nothing holds that count to the network certifier's
count over the span until a composition ties `named` to it. So any other tracked Python file naming either fails here,
unless `ALLOWED` names it. The Lean side's executables are held in that package's suite
(`verity/protocols/accounting/pous/device/lean/tests/test_counted_callers.py`)."""

from __future__ import annotations

import re
from pathlib import Path

from test_repository import ROOT, _tracked

NAMES = ("outcome_counted", "counted_profile")
OWNER = "verity/protocols/accounting/pous/verifier/wipe_period.py"
MIRROR_TESTS = "verity/protocols/accounting/pous/device/lean/tests/"
#: The files that may name them besides `OWNER` and `MIRROR_TESTS`, each with why.
ALLOWED: dict[str, str] = {}
REMEDY = ("the counted check certifies on a count nobody holds until a composition ties `named` to the network "
          "certifier's count: only that composition may call it, and this test's `ALLOWED` names its file then")

USE = re.compile(rf"\b(?:{'|'.join(NAMES)})\b")
HERE = Path(__file__).resolve().relative_to(ROOT).as_posix()


def _callers() -> dict[str, list[str]]:
    found = {}
    for path in _tracked():
        if not path.endswith(".py") or path in (OWNER, HERE) or path.startswith(MIRROR_TESTS):
            continue
        file = ROOT / path
        if file.is_file() and (hits := sorted(set(USE.findall(file.read_text(errors="replace"))))):
            found[path] = hits
    return found


def test_the_counted_check_is_defined_where_this_test_looks() -> None:
    text = (ROOT / OWNER).read_text()
    missing = [n for n in NAMES if not re.search(rf"^def {n}\(", text, re.M)]
    assert not missing, f"{OWNER} defines no {missing}: point this test at the counted check's new home"


def test_no_python_but_the_allowed_calls_the_counted_check() -> None:
    callers = _callers()
    offenders = {p: h for p, h in callers.items() if p not in ALLOWED}
    assert not offenders, "\n".join([f"{p} names {', '.join(h)}" for p, h in offenders.items()] + [f"Remove it: {REMEDY}."])
    stale = [p for p in ALLOWED if p not in callers]
    assert not stale, f"ALLOWED names {stale}, which no longer call the counted check: delete the entries"
