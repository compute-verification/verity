"""RD-24 (note:20261024T2155Z-report-redteam-5): every `Prop` of `Properties.RemoteAudit.Line.Premises` that `LineHeld`
takes as a hypothesis is a named assumption with its own record under `Security/Properties/lean-audit.json`'s
`assumptions` (AGENTS.md "Lean"; `check.py`'s `assumption_records` refuses an unlisted premise). It fails while
`Facts.lean`'s `premisesOf` reads `LineHeld`, a definition outside a `Guarantees` module, with `only`, which keeps only an
assumptions module's constants, so `check` passes with these premises unlisted."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREMISES = ROOT / "Security/Properties/RemoteAudit/Line/Premises.lean"
LINE = ROOT / "Security/Properties/RemoteAudit/Line.lean"
LOCK = ROOT / "Security/Properties/lean-audit.json"
NAMESPACE = "Properties.RemoteAudit.Line"


def prop_defs(text: str) -> set[str]:
    return {m[1] for m in re.finditer(r"^(?:abbrev|def)\s+(\S+)(.*?):=", text, re.MULTILINE | re.DOTALL)
            if re.search(r":\s*Prop\s*$", m[2])}


def statement(text: str, name: str) -> str:
    start = re.search(rf"^def {name} : Prop :=$", text, re.MULTILINE)
    assert start, f"{LINE.relative_to(ROOT)} no longer states `def {name} : Prop :=`"
    end = re.search(r"^(?:/--|def |theorem |end )", text[start.end():], re.MULTILINE)
    return text[start.end():start.end() + (end.start() if end else len(text))]


def test_line_premises_are_listed():
    props, line = prop_defs(PREMISES.read_text()), LINE.read_text()
    listed = set(json.loads(LOCK.read_text())["assumptions"])
    for name in ("LineHeld", "LineHeldIdentity"):
        body = statement(line, name)
        taken = sorted(p for p in props if re.search(rf"\b{re.escape(p)}\b", body))
        unlisted = [f"{NAMESPACE}.{p}" for p in taken if f"{NAMESPACE}.{p}" not in listed]
        assert not unlisted, (f"{name} takes {', '.join(unlisted)} as hypotheses, and the lock's `assumptions` doesn't list "
                              f"them: make `premisesOf` report a hypothesis headed by a constant of a `shape.premises` "
                              f"module under `only`, then `check.py --records Security/Properties --update`")
