"""`verity.ir.refs`: a slice of a reference sequence is the same sequence, in the same representation, however it is found.

* `Concat.slice` (which finds the first part it keeps by bisection) returns what slicing every part in turn returns, by
  representation (`to_json`, the descriptor's encoding), for every range over concatenations of every kind of part,
  zero-length parts and out-of-range bounds included;
* a slice denotes the leaves `list(refs)[lo:hi]`.
"""
from __future__ import annotations

import random

from verity.ir.refs import Affine, Concat, Explicit, Refs, Strided, concat


def _linear(c: Concat, lo: int, hi: int) -> Refs:
    lo, hi = max(lo, 0), min(hi, c.count)
    out = []
    for st, p in zip(c.starts, c.parts):
        a, b = max(lo, st) - st, min(hi, st + p.count) - st
        if b > a:
            out.append(p.slice(a, b))
    return concat(out)


def _part(rng: random.Random) -> Refs:
    k = rng.randrange(5)
    n, b = rng.randrange(0, 7), rng.randrange(0, 50)
    if k == 0:
        return Affine("n", rng.randrange(3), b, n)
    if k == 1:
        return Explicit(tuple(("p", 0, rng.randrange(40)) for _ in range(n)))
    if k == 2:
        inner = rng.randrange(1, 4)
        return Strided("n", rng.randrange(3), b, inner, rng.randrange(0, 9), rng.choice((0, 1, 3)), inner * rng.randrange(0, 4))
    if k == 3:
        return Strided("p", 1, b, 1, 0, 0, n)
    return Concat([_part(rng) for _ in range(rng.randrange(1, 4))])


def test_a_concat_slice_is_the_slice_of_every_part_in_turn():
    rng = random.Random(20261001)
    for _ in range(400):
        c = Concat([_part(rng) for _ in range(rng.randrange(0, 12))])
        leaves = list(c)
        for _ in range(12):
            lo, hi = rng.randrange(-3, c.count + 4), rng.randrange(-3, c.count + 4)
            got = c.slice(lo, hi)
            assert got.to_json() == _linear(c, lo, hi).to_json()
            assert list(got) == leaves[max(lo, 0):max(hi, 0)]


def test_an_affine_slice_clips_to_its_leaves():
    a = Affine("n", 2, 10, 5)
    assert [a.slice(lo, hi).to_json() for lo, hi in ((-2, 3), (1, 9), (4, 2), (5, 5))] == [
        {"affine": ["n", 2, 10, 3]}, {"affine": ["n", 2, 11, 4]}, {"affine": ["n", 2, 14, 0]}, {"affine": ["n", 2, 15, 0]}]
