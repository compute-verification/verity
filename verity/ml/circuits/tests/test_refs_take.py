"""`Coll.take`: member m along an axis is the leaves of every index over the earlier axes, at m, in row-major order, whatever
mix of reference runs holds them."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.refs import Affine, Coll, Explicit, Strided, broadcast, concat
from verity.ml.circuits.types import Array, Tuple, Value


def _member(t, refs, axis, m):
    """The leaves of member ``m`` along ``axis``, from the index arithmetic alone."""
    dims, inner = [], t
    while isinstance(inner, Array):
        dims.append(inner.n)
        inner = inner.elem
    L = inner.leaves
    out = []
    outer = 1
    for d in dims[:axis]:
        outer *= d
    rest = 1
    for d in dims[axis + 1:]:
        rest *= d
    for k in range(outer):
        base = (k * dims[axis] + m) * rest * L
        out += [refs[j] for j in range(base, base + rest * L)]
    return out


def _refs(rng, n):
    parts, left = [], n
    while left:
        c = rng.randint(1, min(left, 9))
        kind = rng.randrange(4)
        if kind == 0:
            parts.append(Affine("n", rng.randrange(5), rng.randrange(50), c))
        elif kind == 1:
            parts.append(Explicit(tuple(("p", rng.randrange(3), rng.randrange(40)) for _ in range(c))))
        elif kind == 2:
            parts.append(broadcast("n", rng.randrange(5), rng.randrange(9), c))
        else:
            parts.append(Strided("n", rng.randrange(5), rng.randrange(30), 3, 7, 2, c))
        left -= c
    return concat(parts)


@pytest.mark.parametrize("seed", range(40))
def test_take_is_the_member_along_its_axis(seed):
    rng = random.Random(seed)
    leaf = rng.choice([Value(1), Value(8), Tuple((Value(1), Value(4)))])
    t = leaf
    for _ in range(rng.randint(2, 4)):
        t = Array(rng.randint(1, 5), t)
    c = Coll(t, _refs(rng, t.leaves))
    depth, inner = 0, t
    while isinstance(inner, Array):
        depth, inner = depth + 1, inner.elem
    for axis in range(depth):
        n = t
        for _ in range(axis):
            n = n.elem
        for m in range(n.n):
            got = c.take(axis, m)
            assert list(got.refs) == _member(t, c.refs, axis, m)
            assert got.refs.count == got.type.leaves
        assert list(c.take(axis, -1).refs) == list(c.take(axis, n.n - 1).refs)
        with pytest.raises(IndexError):
            c.take(axis, n.n)
