"""The remote audit's verifier in Lean (`RemoteAudit`, served by `remote-audit-difftest`) against `audit.Session`, a whole
session at a time: every event's result (the coin released, the append's index and coin, the window's draws, the
verdict, or the rejection with its failure and draw), the outcome and the record. Recorded sessions (`ra_cases.SCENARIOS`)
are honest ones and ones that each reach one of `audit.FAILURES`; drawn ones are those with their events, fields and
parameters edited, which reach the rest of the bookkeeping and the decoders' refusals."""

from __future__ import annotations

import copy
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
def test_a_recorded_session_gives_the_same_results_outcome_and_record(server, name) -> None:
    case, outcome = M.recorded(name)
    assert outcome == M.SCENARIOS[name][3]
    assert D.replay(PORT, case, server)[0] == "output"


def test_the_recorded_sessions_reach_every_failure_but_tile() -> None:
    """`tile` is a registration's that binds PoUW's tiles, which the port refuses."""
    reached = {s[3] for s in M.SCENARIOS.values()}
    assert reached == {*audit.FAILURES, "accepted"} - {"tile"}


def test_a_registration_that_binds_tiles_is_refused_by_both(server) -> None:
    case = copy.deepcopy(M.recorded("honest")[0])
    case["registration"]["bind"] = True
    assert D.replay(PORT, case, server)[0] == "reject"
