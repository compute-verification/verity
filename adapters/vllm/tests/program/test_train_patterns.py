"""`verity_vllm.program.frontend.rules.patterns`' three walks back from a node of the traced graph (`_through_views`,
`_static_positions_source`, `_buffer_source`): each passes a cast only where the trace recorded a value on both of its sides
and the two have one dtype, and the buffer's walk also takes one widening of bf16 to f32."""
from __future__ import annotations

import pytest
import torch

from verity_vllm.program.frontend.rules.patterns import (
    _buffer_source,
    _static_positions_source,
    _through_views,
)

aten = torch.ops.aten


def _cast_of_arange(source: torch.dtype | None, cast: torch.dtype) -> tuple[torch.fx.Node, torch.fx.Node]:
    """An `arange` and the cast that reads it, each with the value a trace records (a meta tensor of its dtype); `source`
    None: none recorded for the `arange`."""
    graph = torch.fx.Graph()
    arange = graph.call_function(aten.arange.default, (4,))
    if source is not None:
        arange.meta["val"] = torch.empty(4, dtype=source, device="meta")
    node = graph.call_function(aten.to.dtype, (arange, cast))
    node.meta["val"] = torch.empty(4, dtype=cast, device="meta")
    return arange, node


def test_a_cast_between_two_values_of_one_dtype_is_passed() -> None:
    arange, cast = _cast_of_arange(torch.int64, torch.int64)
    assert _through_views(cast) is arange
    assert _static_positions_source(cast) is arange
    assert _buffer_source(cast) == (arange, None)


@pytest.mark.parametrize("source", [torch.int64, None], ids=["the dtype changes", "no value recorded for the source"])
def test_any_other_cast_is_where_the_walk_ends(source) -> None:
    _, cast = _cast_of_arange(source, torch.float32)
    assert _through_views(cast) is cast
    assert _static_positions_source(cast) is None
    assert _buffer_source(cast) == (None, None)


def test_the_buffers_walk_takes_one_widening_of_bf16() -> None:
    arange, cast = _cast_of_arange(torch.bfloat16, torch.float32)
    assert _buffer_source(cast) == (arange, cast)
