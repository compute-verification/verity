"""`verity.ml.circuits.intervals` against sets of integers, and the leaf intervals of every reference form against `iter`."""
from __future__ import annotations

import itertools
import random

import pytest

from verity.ml.circuits import intervals as I
from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided


def as_set(iv) -> set[int]:
    return {x for lo, hi in iv for x in range(lo, hi)}


def random_pairs(rng: random.Random, n: int = 40) -> list[tuple[int, int]]:
    out = []
    for _ in range(rng.randrange(0, 6)):
        lo = rng.randrange(0, n)
        out.append((lo, lo + rng.randrange(-1, 8)))
    return out


def test_norm_is_sorted_disjoint_nonadjacent_and_keeps_the_set():
    rng = random.Random(7)
    for _ in range(500):
        raw = random_pairs(rng)
        iv = I.norm(raw)
        assert isinstance(iv, tuple) and as_set(iv) == as_set(raw)
        assert all(lo < hi for lo, hi in iv)
        assert all(a[1] < b[0] for a, b in zip(iv, iv[1:]))  # adjacent runs are merged
    assert I.norm([(3, 5), (5, 7), (0, 0), (9, 8)]) == ((3, 7),)


def test_set_operations_match_python_sets():
    rng = random.Random(11)
    for _ in range(500):
        a, b = I.norm(random_pairs(rng)), I.norm(random_pairs(rng))
        assert as_set(I.isect(a, b)) == as_set(a) & as_set(b)
        n = rng.randrange(0, 50)
        assert as_set(I.complement(I.restrict(a, 0, n), n)) == set(range(n)) - as_set(a)
        assert I.total(a) == len(as_set(a))
        lo = rng.randrange(0, 30)
        hi = lo + rng.randrange(0, 20)
        assert as_set(I.restrict(a, lo, hi)) == {x - lo for x in as_set(a) if lo <= x < hi}
        for x in range(-2, 52):
            assert I.contains(a, x) == (x in as_set(a))
    assert I.full(0) == () and I.full(3) == ((0, 3),)


def test_strided_intervals_cover_exactly_the_referenced_leaves():
    for inner, outer_stride, inner_stride, count in itertools.product((1, 2, 3, 5), (0, 1, 2, 3, 7), (0, 1, 2, 4, 6), (0, 1, 4, 6, 10, 15)):
        r = Strided("n", 0, 3, inner, outer_stride, inner_stride, count)
        got: list[tuple[int, int]] = []
        I.strided_intervals(r, lambda lo, hi: got.append((lo, hi)))
        expect = {leaf for _, _, leaf in r}
        if outer_stride == 0 and inner_stride == 1 and 0 < count < inner:
            assert as_set(got) >= expect, r  # one partial run of a broadcast row is reported as the whole row
        else:
            assert as_set(got) == expect, r


def test_strided_without_structure_is_enumerated_up_to_the_limit(monkeypatch):
    r = Strided("n", 0, 0, 3, 5, 7, 12)  # no interval structure: enumerated leaf by leaf
    got: list[tuple[int, int]] = []
    I.strided_intervals(r, lambda lo, hi: got.append((lo, hi)))
    assert as_set(got) == {leaf for _, _, leaf in r}
    monkeypatch.setattr(I, "_ENUM_LIMIT", 11)
    with pytest.raises(I.IntervalLimit):
        I.strided_intervals(r, lambda lo, hi: None)


def test_ref_intervals_group_by_space_and_index():
    refs = Concat([Affine("p", 0, 2, 3), Explicit((("p", 1, 4), ("n", 2, 0))), Strided("p", 0, 10, 2, 5, 1, 4), Affine("p", 0, 0, 0)])
    out: dict = {}
    I.ref_intervals(refs, out)
    assert {k: I.norm(v) for k, v in out.items()} == {("p", 0): ((2, 5), (10, 12), (15, 17)), ("p", 1): ((4, 5),), ("n", 2): ((0, 1),)}
