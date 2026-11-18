"""The refresh attacks on the slot schedule (`refresh_attacks`): the Python books and the compiled Lean verifier give
each row the same verdict, the honest control accepted with nothing free and every attack refused with the failure its
rule names; with its rule off the books take each attack and credit what it carries; a laundered registered re-key is
caught by the sample, and a skipped one read on is refused at receive."""

from __future__ import annotations

import shutil

import pytest
import refresh_attacks as A
import verity_difftest as D

pytestmark = pytest.mark.slow
lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

server = D.server_fixture(A.R.PORT)

OFF = {"plain stacking": 1, "carrier of a registered chunk": 1, "forged age": 4,
       "laundered origin: registered, m = 2": 2, "laundered origin: re-key of a re-key": "origin-path",
       "early plan": 1, "late re-key": 1, "shifted origin (prefix_origin_reads_key)": 0}
"""Each switched row's free receipts with its rule off, or the failure another rule still names."""


@lake
@pytest.mark.parametrize("row", A.ROWS, ids=[r.name for r in A.ROWS])
def test_the_books_and_the_lean_give_each_row_the_verdict_its_rule_names(server, row: A.Row) -> None:
    py, lean = A.play(server, row.cheat)
    assert A.agree(py, lean), (py, lean)
    assert (lean.outcome, lean.free) == (("accepted", 0) if row.failure is None else (row.failure, None)), lean


@pytest.mark.parametrize("row", [r for r in A.ROWS if r.switched], ids=[r.name for r in A.ROWS if r.switched])
def test_with_its_rule_off_the_books_take_each_attack_and_credit_what_it_carries(row: A.Row) -> None:
    off = A.rule_off(row)
    want = OFF[row.name]
    assert (off.outcome, off.free) == ((want, None) if type(want) is str else ("accepted", want)), off


@lake
def test_a_laundered_rekey_is_caught_by_the_sample_and_earns_its_carried_reads_uncaught(server) -> None:
    """Every chunk laundered is caught under any coin; one chunk is a rate, an uncaught run earning its one carried read,
    no more than the sample term charges."""
    every = A.laundering(server, A.CLASS, 1)
    assert (every.caught, every.differ) == (every.trials, 0)
    one = A.laundering(server, 1, 2)
    assert one.differ == 0 and one.caught + len(one.free) == one.trials and set(one.free) <= {1}
    assert all(f <= c for f, c in zip(one.free, one.sample)) and 0 < one.bound < 1


@lake
def test_a_skipped_registered_rekey_read_on_is_refused_at_receive_under_any_coin(server) -> None:
    r = A.skipping(server, 2)
    assert (r.caught, r.differ) == (r.trials, 0)
