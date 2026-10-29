"""P2's modular squaring circuit (`pous.circuit`) against Python's integers, through `verity`'s bit evaluator."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as EB
from verity.core.protocols.pous import circuit as P

RNG = random.Random(11)


def _to_bits(xs: list[int], n: int) -> np.ndarray:
    return np.array([[(x >> j) & 1 for j in range(n)] for x in xs], dtype=np.uint8)


def _from_bits(arr: np.ndarray) -> list[int]:
    return [int("".join(map(str, row[::-1].tolist())), 2) for row in arr]


def _run(fn, xs: list[int], n: int) -> list[int]:
    return _from_bits(EB.evaluate_bits(fn, _to_bits(xs, n)))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, P.SCHOOL, P.SCHOOL + 1, 48, 49, P.TOOM, P.TOOM + 1, P.TOOM + 2, 257])
def test_square_is_the_integer_square(n):
    """Schoolbook up to `SCHOOL`, Karatsuba (equal and unequal halves) up to `TOOM`, Toom-3 (every N mod 3) above."""
    xs = [0, 1, (1 << n) - 1, 1 << (n - 1)] + [RNG.getrandbits(n) for _ in range(24)]
    assert _run(bind(P.PousSquare, N=n), xs, n) == [x * x for x in xs]


@pytest.mark.parametrize(("w", "c0"), [(30, 5), (64, 59), (128, 21065), (301, 21065), (521, 21065)])
def test_mod_square_is_fully_reduced(w, c0):
    p = (1 << w) - c0
    xs = [0, 1, 2, p - 1, p, p + 1, (1 << w) - 1, 1 << (w - 1), c0, p - c0] + [RNG.randrange(1 << w) for _ in range(24)]
    assert _run(bind(P.PousModSquare, W=w, C0=c0), xs, w) == [x * x % p for x in xs]


@pytest.mark.parametrize(("w", "c0"), [(30, 5), (64, 59), (64, (1 << 32) - 1)])
def test_fold_reduces_every_double_width_input(w, c0):
    """Including the inputs whose first fold carries into the second, and the largest C0 the fold admits."""
    p = (1 << w) - c0
    top = (1 << 2 * w) - 1
    xs = [0, p, 2 * p - 1, (p - 1) ** 2, top, top - (1 << w) + 1, ((1 << w) - 1) << w, c0 << w]
    xs += [RNG.getrandbits(2 * w) for _ in range(24)]
    assert _run(bind(P.PousModFold, W=w, C0=c0), xs, 2 * w) == [x % p for x in xs]


def test_fold_refuses_a_c0_two_folds_cannot_reduce():
    with pytest.raises(ValueError, match="C0"):
        bind(P.PousModFold, W=64, C0=1 << 32).signature()


def test_counts_at_the_circuit_check_binding():
    """The binding `circuit-check` traces; its Flock lowering is pinned in circuit_check's `pins.json`."""
    assert EB.compile(bind(P.PousModSquare, W=128, C0=21065)).counts == {"and": 10636, "xor": 27158, "not": 1086}


SLOW = pytest.mark.skipif(not os.environ.get("VERITY_POUS_SLOW"), reason="minutes at P2's width: VERITY_POUS_SLOW=1")


@SLOW
def test_counts_at_p2():
    """The gate counts the setup cost model reads (`circuit`'s docstring): square, fold and their composition."""
    assert EB.compile(bind(P.PousSquare, N=P.P2_W)).counts == {"and": 13892750, "xor": 37399902, "not": 2488033}
    assert EB.compile(bind(P.PousModFold, W=P.P2_W, C0=P.P2_C0)).counts == {"and": 148102, "xor": 444400, "not": 15}
    assert EB.compile(bind(P.PousModSquare, W=P.P2_W, C0=P.P2_C0)).counts == {"and": 14040852, "xor": 37844302,
                                                                            "not": 2488048}


@SLOW
def test_mod_square_at_p2():
    w, p = P.P2_W, (1 << P.P2_W) - P.P2_C0
    xs = [0, 1, p - 1, p, (1 << w) - 1, 1 << (w - 1)] + [RNG.randrange(p) for _ in range(4)]
    assert _run(bind(P.PousModSquare, W=w, C0=P.P2_C0), xs, w) == [x * x % p for x in xs]
