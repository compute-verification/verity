"""``credited_in_c window``'s order: each range of it is one batch of the window's Program, so a range holds only members
that read the same writers' fields, which is the one form an edge reads a batch in."""
from __future__ import annotations

import credited_in_c as CI
import pytest

SEED, LINE, GRAM = "Seed", "Line", "Gram"


def test_a_batch_whose_members_read_different_fields_of_one_writer_is_a_range_a_field() -> None:
    lines = [(("key", 0, 0), ("index", 0, 2 + m)) for m in range(3)]
    assert CI.order_runs([(SEED, 0, [()]), (LINE, 0, lines)]) == [(SEED, 0, 1), (LINE, 0, 1), (LINE, 1, 1), (LINE, 2, 1)]


def test_members_that_read_one_field_of_each_writer_stay_one_range() -> None:
    same = [(("fa", 1, 0),)] * 4
    assert CI.order_runs([(SEED, 0, [()]), (GRAM, 0, same)]) == [(SEED, 0, 1), (GRAM, 0, 4)]


def test_a_window_whose_reader_has_no_row_past_the_last_block_is_refused_by_name() -> None:
    from verity.circuits.defs import bind
    from verity.circuits.ops.rows import Bf16AddRow
    CI.provable({"r": bind(Bf16AddRow, N=64)}, {})
    with pytest.raises(SystemExit, match="WINDOW-BLOCK: Bf16AddRow_v1{N=1}.*no row past"):
        CI.provable({"r": bind(Bf16AddRow, N=1)}, {})


def test_consecutive_nodes_of_one_template_are_one_range_only_where_they_read_alike() -> None:
    a, b = [(("x", 0, 0),)], [(("x", 0, 1),)]
    assert CI.order_runs([(GRAM, 0, a), (GRAM, 1, a)]) == [(GRAM, 0, 2)]
    assert CI.order_runs([(GRAM, 0, a), (GRAM, 1, b)]) == [(GRAM, 0, 1), (GRAM, 1, 1)]
