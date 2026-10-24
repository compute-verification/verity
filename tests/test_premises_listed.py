"""RD-24 (note:20261024T2155Z-report-redteam-5): every `Prop` of a premises module that a property's statement takes as
a hypothesis is a named assumption with its own record under `Security/Properties/lean-audit.json`'s `assumptions`
(AGENTS.md "Lean"; `check.py`'s `assumption_records` refuses an unlisted premise). It fails while `Facts.lean`'s
`premisesOf` reads a statement that is a definition outside a `Guarantees` module with `only`, which keeps only an
assumptions module's constants, so `check` passes with these premises unlisted: `LineHeld`'s
`Properties.RemoteAudit.Line.Premises` and the network certifier's `NetTiming.DeviceIntact`."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "Security/Properties/lean-audit.json"

#: (premises module, its namespace, the module stating the properties, the properties)
CASES = [
    ("Security/Properties/RemoteAudit/Line/Premises.lean", "Properties.RemoteAudit.Line",
     "Security/Properties/RemoteAudit/Line.lean", ("LineHeld", "LineHeldIdentity")),
    ("Security/Definitions/NetworkCertifier/Premises.lean", "NetTiming",
     "Security/Properties/CertifierNetwork/Runs.lean",
     ("AcceptedRowsAreProgramRows", "AcceptedRunsAreProgramRuns", "EncardAcceptedObservationsLe")),
]


def prop_defs(text: str) -> set[str]:
    return {m[1] for m in re.finditer(r"^(?:abbrev|def)\s+(\S+)(.*?):=", text, re.MULTILINE | re.DOTALL)
            if re.search(r":\s*Prop\s*$", m[2])}


def statement(text: str, where: str, name: str) -> str:
    start = re.search(rf"^def {name} : Prop :=$", text, re.MULTILINE)
    assert start, f"{where} no longer states `def {name} : Prop :=`"
    end = re.search(r"^(?:/--|def |theorem |end )", text[start.end():], re.MULTILINE)
    return text[start.end():start.end() + (end.start() if end else len(text))]


@pytest.mark.parametrize(("premises", "namespace", "where", "names"), CASES, ids=[c[3][0] for c in CASES])
def test_premises_are_listed(premises, namespace, where, names):
    props, text = prop_defs((ROOT / premises).read_text()), (ROOT / where).read_text()
    listed = set(json.loads(LOCK.read_text())["assumptions"])
    for name in names:
        body = statement(text, where, name)
        taken = sorted(p for p in props if re.search(rf"\b{re.escape(p)}\b", body))
        assert taken, f"{name} takes no `Prop` of {premises}: drop it from CASES"
        unlisted = [f"{namespace}.{p}" for p in taken if f"{namespace}.{p}" not in listed]
        assert not unlisted, (f"{name} takes {', '.join(unlisted)} as hypotheses, and the lock's `assumptions` doesn't "
                              f"list them: make `premisesOf` report a hypothesis headed by a constant of a "
                              f"`shape.premises` module under `only`, then `check.py --records Security/Properties "
                              f"--update`")
