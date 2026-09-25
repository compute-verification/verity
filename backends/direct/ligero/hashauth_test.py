"""hashauth.layout_error: a VU's x row and W column are fixed by its y leaf and the committed layout (red-team-standard-hash
R1), the same rule as ligero-verify's ``auth::layout_error``.

    uv run pytest backends/direct/ligero/hashauth_test.py
"""
from __future__ import annotations

from .auth import TreeRef
from .hashauth import HashAuth, layout_error


def _auth(vu, x, w, counts) -> HashAuth:
    trees = tuple(TreeRef(n, bytes(32), o, c, bytes(32)) for n, o, c in zip(("a", "b", "y"), (-1, -2, -1), counts))
    return HashAuth(tuple(vu), tuple(x), tuple(w), trees)


def test_unshared_is_x_equals_w_equals_vu() -> None:
    assert layout_error(_auth([0, 2, 5], [0, 2, 5], [0, 2, 5], (6, 6, 6))) is None
    assert "committed layout" in layout_error(_auth([0, 1], [1, 1], [1, 1], (2, 2, 2)))       # VU 0 on VU 1's (x, W)
    assert "committed layout" in layout_error(_auth([0, 1], [0, 1], [1, 0], (2, 2, 2)))       # W columns swapped


def test_tile_is_row_major_over_the_columns() -> None:
    assert layout_error(_auth([3, 4, 5], [1, 1, 1], [0, 1, 2], (2, 3, 6))) is None             # relchain.tile_indices(2, 3, 3, 6)
    assert "committed layout" in layout_error(_auth([3, 4, 5], [1, 1, 1], [0, 2, 2], (2, 3, 6)))
    assert "committed layout" in layout_error(_auth([3], [0], [1], (3, 2, 6)))


def test_other_shapes_and_ranges_are_refused() -> None:
    assert layout_error(_auth([0], [0], [0], (1, 1, 1))) is None
    assert "neither" in layout_error(_auth([0], [0], [0], (2, 3, 5)))
    assert layout_error(_auth([6], [2], [0], (2, 3, 6))) == "auth: y: leaf index out of range"
    assert layout_error(_auth([], [], [], (0, 0, 0))) is None
