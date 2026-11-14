"""An output field's leaves, each written once by dense reads (`verity.commitments.windows.boundary`): the closed form
against a reference that lists every leaf of every read, on every small read and pair of reads, and the cases
`Spot.BoundaryWritten`'s examples hold."""
from __future__ import annotations

import itertools
import math

import pytest

from verity.commitments.windows import boundary as B

#: four served ids of two 16-bit leaves each, a unit writing one id
IDS = [(0, (4,), (2,), 2)]


def leaves_of(base, radix, step, row) -> list[int]:
    out = []
    for i in range(math.prod(radix)):
        src, q = base, i
        for r, s in zip(reversed(radix), reversed(step)):
            src, q = src + q % r * s, q // r
        out += range(src, src + row)
    return out


def dense(base, radix, step, row) -> bool:
    ls = sorted(leaves_of(base, radix, step, row))
    return (bool(ls) and all(r >= 1 and (r == 1 or s >= 0) for r, s in zip(radix, step))
            and ls == list(range(ls[0], ls[0] + len(ls))))


def reference(leaves, reads) -> bool:
    return (all(dense(*r) for r in reads)
            and sorted(x for r in reads for x in leaves_of(*r)) == list(range(leaves)))


@pytest.mark.parametrize("leaves, reads, want", [
    (8, IDS, True),
    (8, [(0, (3,), (2,), 2)], False),
    (8, [(0, (4,), (0,), 2)], False),
    (8, [(6, (4,), (-2,), 2)], False),
    (8, [(4, (2,), (2,), 2), (0, (2,), (2,), 2)], True),
    (8, [*IDS, (6, (1,), (0,), 2)], False),
    (8, [(0, (5,), (2,), 2)], False),
    (8, [(0, (2,), (4,), 2), (2, (2,), (4,), 2)], False),
    (8, [(0, (2, 2), (4, 2), 2)], True),
    (8, [(0, (2, 2), (2, 4), 2)], True),
    (8, [(0, (1, 4), (-5, 2), 2)], True),
    (0, [(0, (1,), (0,), 0)], False),
    (0, [(0, (0,), (2,), 2)], False),
    (0, [], True),
], ids=["ids", "three of four ids", "fan-out", "negative step", "two halves", "overlap", "past the field",
        "interleaved", "row digit fast", "row digit slow", "radix 1", "row of no leaves", "radix 0", "no reads"])
def test_the_examples(leaves, reads, want):
    assert B.written(leaves, reads) is want and reference(leaves, reads) is want


def _reads(bases, radices, steps, rows):
    for n in (0, 1, 2):
        for ds in itertools.product(itertools.product(radices, steps), repeat=n):
            for base, row in itertools.product(bases, rows):
                yield base, tuple(r for r, _ in ds), tuple(s for _, s in ds), row


def test_the_closed_form_is_the_reference_on_every_small_read():
    """Every read of up to two digits, radix 0 to 3, step -2 to 4, base -1 to 2 and row 0 to 2, alone and against
    fields of 0 to 12 leaves; and every pair of reads of one digit from a smaller grid: the closed form decides each as
    the reference does, and its extent is the reference's least leaf and length for exactly the dense reads."""
    singles = list(_reads(range(-1, 3), range(4), range(-2, 5), range(3)))
    for r in singles:
        want = (min(leaves_of(*r)), min(leaves_of(*r)) + len(leaves_of(*r))) if dense(*r) else None
        assert B.extent(*r) == want, r
        for leaves in range(13):
            assert B.written(leaves, [r]) == reference(leaves, [r]), (leaves, r)
    small = [r for r in _reads(range(0, 5), range(1, 4), range(0, 4), range(1, 3)) if len(r[1]) <= 1]
    for a, b in itertools.product(small, repeat=2):
        for leaves in (4, 6, 8):
            assert B.written(leaves, [a, b]) == reference(leaves, [a, b]), (leaves, a, b)
