"""`verity.circuits.partition_v2`: the public grid and the public size classes."""
from __future__ import annotations

import pytest

from verity.circuits import partition_v2 as V


def test_the_grid():
    assert [V.grid(x) for x in (0, 1, 15, 16, 17, 31, 32, 33, 2049)] == [0, 1, 15, 16, 18, 32, 32, 36, 2304]
    assert V.grid(11, 2) == 12 and V.grid(2304) == 2304
    for x in range(1, 5000):
        y = V.grid(x)
        assert x <= y < x * (1 + 2 ** (1 - V.GRID_BITS)) and V.grid(y) == y
    with pytest.raises(ValueError):
        V.grid(-1)


def test_the_size_classes():
    assert V.SIZE_CLASSES[0] == 16 and V.SIZE_CLASSES[-1] == V.SIZE_CLASS_MAX
    assert [V.size_class(b) for b in (1, 16, 17, 80, V.SIZE_CLASS_MAX, V.SIZE_CLASS_MAX + 1)] == [16, 16, 32, 128, V.SIZE_CLASS_MAX, None]
