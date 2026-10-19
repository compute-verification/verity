"""`audit.Session` held to the remote audit's verifier in Lean (`RemoteAudit.Session`, the definition, served by
`remote-audit-difftest`), a whole session at a time: every event's result (the coin released, the append's index and
coin, the window's draws, the verdict, or the rejection with its failure and draw), the outcome and the record.
Recorded sessions (`ra_cases.SCENARIOS`) are honest ones and ones that each reach one of `audit.FAILURES` in the Lean;
drawn ones are those with their events, fields and parameters edited, which reach the rest of the bookkeeping and the
decoders' refusals."""

from __future__ import annotations

import copy
import json
import shutil

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import audit

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PORT = D.Port(M.EXE, "session", M.session, st.nothing(), package=M.PACKAGE)
server = D.server_fixture(PORT)


@pytest.mark.parametrize("name", list(M.SCENARIOS))
def test_audit_py_gives_a_recorded_session_the_leans_results_outcome_and_record(server, name) -> None:
    case, _ = M.recorded(name)
    kind, lean = D.replay(PORT, case, server)
    assert kind == "output" and json.loads(lean)["outcome"] == M.SCENARIOS[name][3]


def test_the_recorded_sessions_reach_every_failure_but_tile() -> None:
    """`tile` is a registration's that binds PoUW's tiles, which the Lean refuses."""
    reached = {s[3] for s in M.SCENARIOS.values()}
    assert reached == {*audit.FAILURES, "accepted"} - {"tile"}


def test_a_session_that_defers_residency_at_a_lifetime_of_one_is_refused_by_both(server) -> None:
    """Deferred, a block appended in window w is resident from w or w + 1, so its re-key falls at age L − 1: both sides
    refuse the parameters at L = 1, and take them at L = 2."""
    case = copy.deepcopy(M.recorded("honest-rekey")[0])
    case["params"]["deferred"] = True
    assert D.replay(PORT, case, server)[0] == "reject"
    case = copy.deepcopy(M.recorded("honest")[0])
    case["params"] |= {"deferred": True, "lifetime": 2}
    assert D.replay(PORT, case, server)[0] == "output"


def test_a_registration_that_binds_tiles_is_refused_by_both(server) -> None:
    case = copy.deepcopy(M.recorded("honest")[0])
    case["registration"]["bind"] = True
    assert D.replay(PORT, case, server)[0] == "reject"
