"""`HonestLogAccepted` (`security/Properties/Poub/Remote/HonestLog.lean`), the premise the honest witnesses of
`LineHeld` and `LineHeldAtSubprotocol` take, held to what the compiled Lean and the Python verifier do: the module the
premise reads is `honest_log.py`'s rendering of the honest one-window period's case; `poub-protocol`
(`Poub.Protocol.check`) accepts that case with the live session's public outputs and profile, as the Python verifier
does; and `poub-difftest`'s session of it (`Poub.Session`) samples the Python verifier's receipt at the window's close."""

from __future__ import annotations

import json
import shutil

import honest_log as H
import pytest
import ra_cases as M
import verity_difftest as D
from hypothesis import strategies as st
from test_protocol import PORT, protocol

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SESSION = D.Port(M.EXE, "session", M.session, st.nothing(), package=M.PACKAGE, of_record=M.OF_RECORD)


def test_the_premises_module_is_the_case() -> None:
    """Regenerate it with `python honest_log.py` from this directory."""
    assert H.MODULE.read_text() == H.lean_module(H.run().case)


def test_the_compiled_check_accepts_the_case_with_the_live_answer() -> None:
    run = H.run()
    assert run.answer["verdict"] == "accept"
    assert protocol(run.case) == run.answer
    _, text = D.replay(PORT, run.case)
    lean = json.loads(text)
    assert (lean["verdict"], lean["public"], lean["profile"]) == ("accept", run.answer["public"], run.answer["profile"])


def test_the_sessions_close_samples_the_one_receipt() -> None:
    case = H.run().case
    evidence = json.loads(bytes.fromhex(case["evidence"]))
    session = {"registration": json.loads(bytes.fromhex(case["commitment"])), **evidence, **case["params"]}
    _, text = D.replay(SESSION, session)
    lean = json.loads(text)
    kinds = [e["kind"] for e in evidence["events"]]
    assert kinds == ["start", "receive", "boundary", "close", "verify"]
    assert lean["outcome"] == "accepted"
    assert lean["results"][kinds.index("close")] == {"draws": [[0, 0]]}
