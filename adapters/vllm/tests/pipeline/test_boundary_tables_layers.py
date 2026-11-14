"""`boundary_tables.layers_of`: the history tables' layers from a serve's configuration, a count or the layers themselves."""
from __future__ import annotations

import pytest

from verity_vllm.pipeline import boundary_tables as BT


def test_a_count_is_the_first_that_many_layers():
    assert BT.layers_of(3) == (0, 1, 2)
    assert BT.layers_of(0) == ()


def test_the_layers_themselves_are_kept_in_their_order():
    assert BT.layers_of([3, 61]) == (3, 61)
    assert BT.layers_of((61, 3)) == (61, 3)
    assert BT.layers_of([]) == ()


@pytest.mark.parametrize("flag", [True, False])
def test_a_flag_is_neither_a_count_nor_the_layers(flag):
    with pytest.raises(TypeError, match="a count or the layers themselves, and a bool is neither"):
        BT.layers_of(flag)
