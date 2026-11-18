from __future__ import annotations

import pytest

from verity.circuits.refs import Affine, Explicit, Grid, Strided, recut


def test_a_reader_whose_width_divides_the_registered_width_reads_q_leaves_a_registered_leaf():
    assert recut(16, 8) == 2 and recut(16, 1) == 16 and recut(32, 32) == 1
    assert recut(16, 8, Affine("n", 0, 4, 6), 5) == 2
    assert recut(16, 8, Grid(2, (("n", 0, 0, 1, ((2, 4),)), ("n", 0, 2, 1, ((2, 4),)))), 4) == 2
    assert recut(16, 8, Strided("n", 0, 0, 4, 8, 1, 8), 8) == 2


@pytest.mark.parametrize("args, name", [
    ((16, 32), "recut-width"),
    ((16, 6), "recut-width"),
    ((16, 8, Affine("n", 0, 1, 2), 4), "recut-boundary"),
    ((16, 8, Affine("n", 0, 0, 3), 4), "recut-boundary"),
    ((16, 8, Strided("n", 0, 0, 3, 4, 1, 6), 8), "recut-boundary"),
    ((16, 8, Strided("n", 0, 0, 2, 3, 1, 4), 8), "recut-boundary"),
    ((16, 8, Strided("n", 0, 0, 2, 4, 2, 4), 8), "recut-boundary"),
    ((16, 8, Explicit((("n", 0, 0),)), 4), "recut-boundary"),
    ((16, 8, Affine("n", 0, 6, 4), 4), "recut-overrun"),
    ((16, 8, Strided("n", 0, 0, 4, 8, 1, 8), 5), "recut-overrun"),
])
def test_a_read_not_of_whole_registered_leaves_inside_its_node_is_refused_by_name(args, name):
    with pytest.raises(ValueError, match=name):
        recut(*args)
