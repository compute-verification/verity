"""training_slowdown.model: the model's constants are the same numbers in every process."""
import math

import pytest
import torch
from synthetic import SHAPE
from torch.utils._python_dispatch import TorchDispatchMode

from training_slowdown import starts
from training_slowdown.model import LM, ROPE_THETA, Size


class Taken(TorchDispatchMode):
    """How many numbers each cos and each sin computed under it was taken of."""

    def __init__(self):
        super().__init__()
        self.of: list[int] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket in (torch.ops.aten.cos, torch.ops.aten.sin):
            self.of.append(args[0].numel())
        return func(*args, **(kwargs or {}))


def tables() -> list[tuple[int, int]]:
    """The context and head width of the tests' start and of every registered one: what a rotary table's shape is."""
    shapes = [SHAPE, *(starts.start(name).shape for name in starts.registry()["starts"])]
    return sorted({(s["ctx"], s["d_model"] // s["n_head"]) for s in shapes})


@pytest.mark.parametrize("ctx, hd", tables())
def test_the_rotary_tables_are_made_one_position_at_a_time_and_are_right_to_the_last_place(ctx, hd):
    """torch cuts a cos or a sin of more than a few thousand numbers across its threads, and a thread's share of the
    first one a process computes can come back at low accuracy, the same input giving another table in the next
    process. A model whose tables were made that way is another function from process to process, so the model takes
    each cos and sin of one position's row, which torch computes in the calling thread. The tables themselves are
    held to the cosine and sine of each angle in double precision, to float32's last place: what a low-accuracy share
    misses by three orders of magnitude, and a bound no thread count or CPU moves."""
    with Taken() as taken:
        m = LM(Size(hd, 1, 1, 1), vocab=1, ctx=ctx)
    assert taken.of == [hd // 2] * (2 * ctx)
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    angles = torch.outer(torch.arange(ctx, dtype=torch.float32), freqs)
    for table, exact in ((m.cos, math.cos), (m.sin, math.sin)):
        assert table.shape == (ctx, hd // 2) and table.dtype == torch.float32
        want = torch.tensor([exact(a) for a in angles.flatten().tolist()], dtype=torch.float64).view_as(angles)
        assert (table.double() - want).abs().max() <= torch.finfo(torch.float32).eps
