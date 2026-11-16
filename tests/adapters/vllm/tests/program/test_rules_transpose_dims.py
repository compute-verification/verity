"""`vllm_dsa_absorb._transpose_01` reads a transpose's two dimensions through `int()`: what `int()` takes names a
dimension, and what it refuses raises `int()`'s own error."""
from __future__ import annotations

import pytest
import torch
import torch.fx as fx

from verity_vllm.program.frontend.rules.vllm_dsa_absorb import _transpose_01


class _Shapes:
    """In the derive context's place: every node has the one shape."""

    def __init__(self, shape: tuple[int, ...] | None) -> None:
        self.shape = shape

    def shape_of_node(self, node):
        return self.shape


def _transpose(d0, d1) -> fx.Node:
    graph = fx.Graph()
    return graph.call_function(torch.ops.aten.transpose.int, (graph.placeholder("x"), d0, d1))


@pytest.mark.parametrize(("dims", "swaps"), [
    ((0, 1), True), ((1, 0), True), ((-3, 1), True), ((-3, -2), True), ((True, False), True),
    ((0, 2), False), ((1, 2), False), ((2, 0), False),
    ((1.0, 0.0), True), ((0.5, 1), True), (("0", "1"), True),
], ids=repr)
def test_a_transpose_swaps_the_first_two_of_three_dimensions(dims, swaps):
    assert _transpose_01(_transpose(*dims), _Shapes((2, 3, 4))) is swaps


@pytest.mark.parametrize(("dims", "error"), [(("a", 1), ValueError), ((None, 1), TypeError), ((0, None), TypeError)], ids=repr)
def test_a_dimension_int_refuses_raises_its_error(dims, error):
    with pytest.raises(error):
        _transpose_01(_transpose(*dims), _Shapes((2, 3, 4)))


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4, 5), None], ids=repr)
def test_no_dimension_is_read_of_a_tensor_that_has_not_three(shape):
    assert _transpose_01(_transpose(None, "a"), _Shapes(shape)) is False
