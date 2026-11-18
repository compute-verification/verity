"""The refresh attacks given their verdicts by the compiled Lean verifier on the slot schedule (`refresh_attacks_lean`):
the honest control is accepted and earns nothing free; every attack is refused with the failure its rule names; a
laundered registered re-key is caught by the sample, and a skipped one read on is refused at receive."""

from __future__ import annotations

import shutil

import pytest
import refresh_attacks_lean as A
import verity_difftest as D

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]

server = D.server_fixture(A.R.PORT)


@pytest.mark.parametrize("row", A.ROWS, ids=[r.name for r in A.ROWS])
def test_each_row_has_the_lean_verdict_its_rule_names(server, row: A.Row) -> None:
    v = A.play(server, row.cheat)
    if row.failure is None:
        assert (v.outcome, v.free) == ("accepted", 0), v
    else:
        assert v.outcome == row.failure, v


def test_a_laundered_rekey_is_caught_by_the_sample_and_earns_its_carried_reads_uncaught(server) -> None:
    """Every chunk laundered is caught under any coin; one chunk is a rate, an uncaught run earning its one carried read
    (R1@1's key ends at window 3), no more than the sample term charges."""
    every = A.laundering(server, A.CLASS, 1)
    assert every.caught == every.trials
    one = A.laundering(server, 1, 2)
    assert one.caught + len(one.free) == one.trials and set(one.free) <= {1}
    assert all(f <= c for f, c in zip(one.free, one.sample)) and 0 < one.bound < 1


def test_a_skipped_registered_rekey_read_on_is_refused_at_receive_under_any_coin(server) -> None:
    r = A.skipping(server, 2)
    assert r.caught == r.trials
