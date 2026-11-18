"""`wiring.Wired`'s closed form: unstrided, the reader's units in a run; with reader steps, several pieces on one port, each
reader unit read by exactly one."""

from __future__ import annotations

import numpy as np
import pytest

from verity.commitments.periods import wiring as WI


def test_an_unstrided_wire_reads_its_closed_form() -> None:
    w = WI.Wired("x", "a", "o0", 2, (4,), 7, (-2,))
    assert w.positions(2, 4).tolist() == [7, 5, 3, 1]
    assert w.units(0, 8)[0].tolist() == [2, 3, 4, 5]
    assert w.units(0, 8)[1].tolist() == [7, 5, 3, 1]


def test_pieces_strided_on_the_readers_side_wire_one_port_span_by_span() -> None:
    """Two spans of two reader units over spans of four writer leaves: in span s, unit 0 reads leaf 4 s + 3, unit 1 leaf
    4 s; each piece is one in-span run, the span its repeat digit."""
    first = WI.Wired("x", "a", "o0", 0, (2, 1), 3, (4, 0), rstep=(2, 0))
    second = WI.Wired("x", "a", "o0", 1, (2, 1), 0, (4, 0), rstep=(2, 0))
    assert WI.reads((first, second), 0, 4).tolist() == [3, 0, 7, 4]
    assert WI.reads((first, second), 1, 2).tolist() == [0, 7]
    assert WI.Wired.of(first.doc()) == first and "rstep" not in WI.Wired("x", "a", "o0", 0, (2,), 0, (1,)).doc()


def test_a_unit_read_by_no_piece_or_by_two_is_refused() -> None:
    first = WI.Wired("x", "a", "o0", 0, (2, 1), 3, (4, 0), rstep=(2, 0))
    with pytest.raises(ValueError, match="not one each"):
        WI.reads((first,), 0, 4)
    with pytest.raises(ValueError, match="not one each"):
        WI.reads((first, first), 0, 4)


def test_reader_steps_take_one_a_digit_and_no_cut() -> None:
    with pytest.raises(ValueError, match="reader steps"):
        WI.Wired("x", "a", "o0", 0, (2, 2), 0, (1, 1), rstep=(1,))
    with pytest.raises(ValueError, match="reader steps"):
        WI.Wired("x", "a", "o0", 0, (4,), 0, (1,), cut=(0, 2), rstep=(1,))
    assert np.array_equal(WI.Wired("x", "a", "o0", 0, (4,), 0, (1,), rstep=(1,)).positions(0, 4), np.arange(4))
